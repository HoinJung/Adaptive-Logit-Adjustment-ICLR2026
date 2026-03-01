# qwen_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation.logits_process import LogitsProcessor
from torch.nn import CrossEntropyLoss
import copy

# Import Qwen2.5-VL if available
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN2_5_AVAILABLE = True
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    QWEN2_5_AVAILABLE = False

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, drop_rate):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(drop_rate)
        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        if self.need_proj:
            identity = self.proj(identity)
        return out + identity

def adaptor(input_dim, hidden_dim, output_dim, depth=3, decoder=False, residual=True):
    activation = nn.LeakyReLU(0.3)
    drop_rate = 0.3
    layers = []
    curr_dim = input_dim
    for _ in range(depth - 1):
        if residual:
            layers.append(ResidualBlock(curr_dim, hidden_dim, activation, drop_rate))
        else:
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.3),
                nn.Dropout(drop_rate),
            ])
        curr_dim = hidden_dim
    layers.append(nn.Linear(curr_dim, output_dim))
    if decoder:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class DEARAdaptor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, depth=2):
        super().__init__()
        self.adaptor = adaptor(input_dim, hidden_dim, input_dim, depth=depth, residual=True)
    
    def forward(self, x):
        return self.adaptor(x)


class LogitAdjustmentProcessor(LogitsProcessor):
    """
    LogitsProcessor that applies logit adjustment based on token bias and text classifier.
    Adapted from ClipCAP implementation.
    """
    def __init__(
        self,
        token_bias: dict,
        text_classifier,
        tokenizer,
        alpha_tokenizer,
        lam: float,
        neutral: bool,
        s_scale: float,
        threshold: float,
        device: str,
        generated_sequence: Optional[List[int]] = None
    ):
        self.token_bias = token_bias
        self.text_classifier = text_classifier
        self.tokenizer = tokenizer
        self.alpha_tokenizer = alpha_tokenizer
        self.lam = lam
        self.neutral = neutral
        self.s_scale = s_scale
        self.threshold = threshold
        self.device = device
        self.generated_sequence = generated_sequence if generated_sequence is not None else []
        
        # Pre-compute token_id -> beta_value mapping for fast lookup
        # This avoids iterating through entire vocabulary at every generation step
        self.token_id_to_beta = {}
        self.token_ids_with_bias = []
        if token_bias and tokenizer is not None:
            print("Pre-computing token_id to beta mapping for fast logit adjustment...")
            tokenizer_vocab_size = len(tokenizer)
            for token_id in range(tokenizer_vocab_size):
                try:
                    token_str = tokenizer.convert_ids_to_tokens(token_id)
                    if token_str in token_bias:
                        beta_value = token_bias[token_str]
                        if abs(beta_value) >= threshold:  # Only store tokens above threshold
                            self.token_id_to_beta[token_id] = beta_value
                            self.token_ids_with_bias.append(token_id)
                except (IndexError, ValueError):
                    continue
            print(f"Pre-computed {len(self.token_ids_with_bias)} tokens with bias >= {threshold} (out of {tokenizer_vocab_size} total)")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply logit adjustment at each generation step.
        Adapted from ClipCAP: check likely-to-be-selected token's beta value.
        
        Args:
            input_ids: Current input sequence
            scores: Logits for next token prediction [batch_size, vocab_size]
        
        Returns:
            Adjusted logits
        """
        # Get tokenizer vocabulary size
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = scores.shape[-1]
        
        # Find the most likely token (what would be selected)
        # This approximates ClipCAP's approach of checking the selected token
        top_token_id = torch.argmax(scores, dim=-1).item()
        
        # Check if token_id is valid for tokenizer
        if top_token_id >= tokenizer_vocab_size:
            # Token ID is out of range for tokenizer, skip adjustment
            return scores
        
        try:
            token_str = self.tokenizer.convert_ids_to_tokens(top_token_id)
        except (IndexError, ValueError):
            # Token ID is invalid, skip adjustment
            return scores
        
        # Get beta value for the most likely token
        beta_value = 0.0
        if token_str in self.token_bias:
            beta_value = self.token_bias[token_str]
        
        # Check if we should run classifier (threshold check)
        # Similar to ClipCAP: skip_classifier = (abs(beta_value) < threshold)
        skip_classifier = (abs(beta_value) < self.threshold)
        
        if not skip_classifier:
            # Prepare sequence for classifier
            # We'll use the current sequence + the most likely next token
            # This approximates ClipCAP's approach
            classifier_input_ids = input_ids.clone()
            # Append the most likely token for classifier input
            next_token_tensor = torch.tensor([[top_token_id]], device=self.device)
            classifier_input_ids = torch.cat([classifier_input_ids, next_token_tensor], dim=1)
            
            # Ensure proper attention mask
            attention_mask = torch.ones_like(classifier_input_ids, device=self.device)
            
            # Get classifier output
            with torch.no_grad():
                classifier_output = self.text_classifier(
                    input_ids=classifier_input_ids,
                    attention_mask=attention_mask
                )
            
            alpha = classifier_output["logits"].squeeze()
            
            if self.neutral:
                s = 0
                if alpha.dim() > 0:
                    alpha = alpha[1] if len(alpha) > 1 else alpha[0]
                alpha = 2 * torch.sigmoid(alpha) - 1
                alpha = abs(alpha)
            else:
                s = self.s_scale
                if alpha.dim() > 0:
                    alpha = alpha[1] if len(alpha) > 1 else alpha[0]
                alpha = 2 * torch.sigmoid(alpha) - 1
            
            # Apply adjustment to tokens with bias (optimized: only iterate over pre-computed tokens)
            # Formula from ClipCAP: logits[token_id] -= lam * (alpha - s) * beta_i
            adjustment = self.lam * (alpha - s)
            for token_id in self.token_ids_with_bias:
                if token_id < model_vocab_size:  # Ensure token_id is within model vocab range
                    beta_i = self.token_id_to_beta[token_id]
                    if self.neutral:
                        beta_i = abs(beta_i)
                    scores[:, token_id] -= adjustment * beta_i
        
        return scores


class MultiClassRaceLogitProcessor(LogitsProcessor):
    """
    LogitsProcessor for multi-class race detection using hard-coded race tokens.
    Adjusts logits based on image classifier predictions: boosts tokens for predicted race,
    suppresses tokens for other races.
    """
    def __init__(
        self,
        race_probs: torch.Tensor,  # Shape: (7,) - probabilities for 7 race classes
        tokenizer,
        lam: float,
        device: str,
        race_map: dict = None
    ):
        """
        Args:
            race_probs: Probabilities for 7 race classes [East Asian, Indian, Black, White, Middle Eastern, Latino_Hispanic, Southeast Asian]
            tokenizer: Tokenizer to convert token IDs to strings
            lam: Lambda parameter for adjustment strength
            device: Device to use
            race_map: Mapping from race names to class indices
        """
        self.race_probs = race_probs  # Shape: (7,)
        self.tokenizer = tokenizer
        self.lam = lam
        self.device = device
        
        # Race mapping: {'East Asian': 0, 'Indian': 1, 'Black': 2, 'White': 3, 'Middle Eastern': 4, 'Latino_Hispanic': 5, 'Southeast Asian': 6}
        if race_map is None:
            self.race_map = {
                'East Asian': 0,
                'Indian': 1,
                'Black': 2,
                'White': 3,
                'Middle Eastern': 4,
                'Latino_Hispanic': 5,
                'Southeast Asian': 6
            }
        else:
            self.race_map = race_map
        
        # Hard-coded race token mappings (case-insensitive matching)
        # These tokens should correspond to race names in the vocabulary
        self.race_tokens = {
            # East Asian (class 0)
            0: ['asian', 'east', 'eastern', 'chinese', 'japanese', 'korean', 'oriental'],
            # Indian (class 1)
            1: ['indian', 'india', 'south', 'asian'],
            # Black (class 2)
            2: ['black', 'african', 'afro'],
            # White (class 3)
            3: ['white', 'caucasian', 'european'],
            # Middle Eastern (class 4)
            4: ['middle', 'eastern', 'arab', 'arabic', 'persian', 'iranian', 'turkish'],
            # Latino_Hispanic (class 5)
            5: ['latino', 'hispanic', 'latin', 'mexican', 'spanish', 'brazilian'],
            # Southeast Asian (class 6)
            6: ['southeast', 'vietnamese', 'thai', 'filipino', 'indonesian', 'malaysian']
        }
        
        # Build token ID to race class mapping
        self.token_to_race = {}  # Maps token_id -> list of race class indices
        self._build_token_mapping()
    
    def _build_token_mapping(self):
        """Build mapping from token IDs to race classes."""
        vocab_size = len(self.tokenizer)
        for token_id in range(vocab_size):
            try:
                token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                if token_str is None:
                    continue
                # Normalize token string (remove special characters, lowercase)
                token_normalized = token_str.lower().strip('Ġ▁')
                
                # Check which race classes this token belongs to
                race_classes = []
                for race_class, tokens in self.race_tokens.items():
                    for race_token in tokens:
                        if race_token in token_normalized or token_normalized in race_token:
                            race_classes.append(race_class)
                            break
                
                if race_classes:
                    self.token_to_race[token_id] = race_classes
            except (IndexError, ValueError, TypeError):
                continue
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply logit adjustment for multi-class race detection.
        
        For each token:
        - If token corresponds to race class i: boost by lam * prob[i]
        - If token corresponds to other race classes j: suppress by lam * prob[j]
        
        Args:
            input_ids: Current input sequence
            scores: Logits for next token prediction [batch_size, vocab_size]
        
        Returns:
            Adjusted logits
        """
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = scores.shape[-1]
        vocab_range = min(tokenizer_vocab_size, model_vocab_size)
        
        # Convert race_probs to tensor if it's numpy array
        if not isinstance(self.race_probs, torch.Tensor):
            race_probs = torch.tensor(self.race_probs, device=self.device, dtype=scores.dtype)
        else:
            race_probs = self.race_probs.to(device=self.device, dtype=scores.dtype)
        
        # Adjust logits for each token
        for token_id in range(vocab_range):
            if token_id not in self.token_to_race:
                continue
            
            race_classes = self.token_to_race[token_id]
            
            # Calculate adjustment: boost for matching races, suppress for others
            adjustment = 0.0
            for race_class in race_classes:
                # Boost tokens for this race class
                adjustment += self.lam * race_probs[race_class]
            
            # Suppress tokens for other race classes (weighted by their probabilities)
            for other_class in range(len(race_probs)):
                if other_class not in race_classes:
                    adjustment -= self.lam * race_probs[other_class] * 0.1  # Smaller suppression factor
            
            # Apply adjustment
            scores[:, token_id] += adjustment
        
        return scores

def _is_qwen2_5_config(config):
    """Check if config is for Qwen2.5-VL."""
    config_type = type(config).__name__
    if 'Qwen2_5' in config_type:
        return True
    if hasattr(config, 'model_type') and 'qwen2_5_vl' in str(config.model_type):
        return True
    return False

# Create a factory function to get the right class
def create_custom_qwen_class(config):
    """Create CustomQwenForConditionalGeneration with the correct base class."""
    if _is_qwen2_5_config(config):
        if not QWEN2_5_AVAILABLE or Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2.5-VL model requires transformers >= 4.49.0 with Qwen2_5_VLForConditionalGeneration support")
        base_class = Qwen2_5_VLForConditionalGeneration
    else:
        base_class = Qwen2VLForConditionalGeneration
    
    class CustomQwenForConditionalGeneration(base_class):
        def __init__(self, config):
            super().__init__(config)
            self.interim_hidden_state = None
        
        # Copy all methods from the original CustomQwenForConditionalGeneration
        # (forward, generate, etc. will be added below)
    
    return CustomQwenForConditionalGeneration

# For backward compatibility, create a class that works with both
# Use __new__ to create the correct class instance with all methods
def _create_custom_qwen_impl_class(base_class):
    """Create CustomQwen implementation class with all methods."""
    class CustomQwenImpl(base_class):
        def __init__(self, config):
            super(CustomQwenImpl, self).__init__(config)
            self.interim_hidden_state = None
    return CustomQwenImpl

class CustomQwenForConditionalGeneration:
    def __new__(cls, config):
        # Check if this is Qwen2.5-VL config
        if _is_qwen2_5_config(config):
            if not QWEN2_5_AVAILABLE or Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError("Qwen2.5-VL model requires transformers >= 4.49.0 with Qwen2_5_VLForConditionalGeneration support")
            base_class = Qwen2_5_VLForConditionalGeneration
        else:
            base_class = Qwen2VLForConditionalGeneration
        
        # Create the implementation class
        CustomQwenImpl = _create_custom_qwen_impl_class(base_class)
        
        # Copy all methods from this class to the implementation
        for name, method in cls.__dict__.items():
            if callable(method) and name not in ['__new__', '__init__']:
                setattr(CustomQwenImpl, name, method)
        
        # Create instance
        instance = CustomQwenImpl.__new__(CustomQwenImpl, config)
        CustomQwenImpl.__init__(instance, config)
        return instance
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Debiasing parameters
        text_important_indices=None,
        text_mean_features_lowconfidence=None,
        mode=None,
        g_value=0,
        text_classifier=None,
        token_bias=None,
        lam=None,
        neutral=None,
        dear_adaptor=None,
        decoder_mean_features_lowconfidence=None,
        decoder_important_indices=None,
        vqa_name=None,
        **kwargs  # Accept any additional arguments from transformers
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Apply DEAR adaptor if provided
        if dear_adaptor is not None and pixel_values is not None:
            with torch.no_grad():
                # For Qwen2-VL, we need to get vision features differently
                # This is a simplified approach - adjust based on actual Qwen2-VL architecture
                if hasattr(self, 'visual') and self.visual is not None:
                    vision_outputs = self.visual(pixel_values)
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_features = vision_outputs.last_hidden_state.mean(dim=1)  # Pool features
                    else:
                        vision_features = vision_outputs.mean(dim=1)
                    adapted_features = dear_adaptor(vision_features)
        
        # Call parent forward method
        outputs = super(type(self), self).forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs  # Pass through any additional arguments
        )
        
        # Apply debiasing modifications to hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Get last layer hidden states
            
            if mode in ['sfid', 'clipclip'] and decoder_important_indices is not None and decoder_mean_features_lowconfidence is not None:
                # Apply SFID/CLIP-CLIP modifications
                hidden_dim_size = hidden_states.shape[1]
                decoder_mean_features_lowconfidence = decoder_mean_features_lowconfidence.unsqueeze(0).unsqueeze(1)
                decoder_mean_features_lowconfidence = decoder_mean_features_lowconfidence.expand(1, hidden_dim_size, hidden_states.shape[-1])
                hidden_states[:, :, decoder_important_indices] = decoder_mean_features_lowconfidence[:, :, decoder_important_indices]
            
            # Store modified hidden states
            self.interim_hidden_state = hidden_states
        
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        **kwargs
    ):
        """
        Override generate method to support logit adjustment.
        Similar to LLaVA implementation.
        """
        # Extract custom debiasing arguments
        mode = kwargs.pop('mode', None)
        s_scale = kwargs.pop('s_scale', None)
        text_classifier = kwargs.pop('text_classifier', None)
        token_bias = kwargs.pop('token_bias', None)
        lam = kwargs.pop('lam', None)
        neutral = kwargs.pop('neutral', None)
        vqa_tokenizer = kwargs.pop('vqa_tokenizer', None)
        alpha_tokenizer = kwargs.pop('alpha_tokenizer', None)
        threshold = kwargs.pop('threshold', 0.1)
        device = kwargs.pop('device', 'cuda:0')
        
        # For logit mode, create and add logits processor
        # Check if this is intersection mode, race detection (multi-class), or gender detection (binary)
        is_intersection_mode = mode == 'logit' and 'gender_text_classifier' in kwargs and 'race_text_classifier' in kwargs
        is_race_mode = mode == 'logit' and token_bias is None and text_classifier is None and 'race_probs' in kwargs
        is_gender_mode = mode == 'logit' and token_bias is not None and text_classifier is not None and not is_intersection_mode
        
        if is_intersection_mode:
            # Intersection mode: both gender and race debiasing
            from transformers.generation.logits_process import LogitsProcessorList
            from llava_model import IntersectionLogitProcessor
            
            gender_text_classifier = kwargs.pop('gender_text_classifier')
            race_text_classifier = kwargs.pop('race_text_classifier')
            gender_token_bias = kwargs.pop('gender_token_bias', {})
            race_token_bias = kwargs.pop('race_token_bias', {})
            s_scale_gender = kwargs.pop('s_scale_gender', None)
            s_scale_race = kwargs.pop('s_scale_race', None)
            lam_gender = kwargs.pop('lam_gender', None)
            lam_race = kwargs.pop('lam_race', None)
            
            logit_processor = IntersectionLogitProcessor(
                gender_token_bias=gender_token_bias,
                race_token_bias=race_token_bias,
                gender_text_classifier=gender_text_classifier,
                race_text_classifier=race_text_classifier,
                tokenizer=vqa_tokenizer,
                alpha_tokenizer=alpha_tokenizer,
                lam=lam if lam is not None else 0.9,
                lam_gender=lam_gender,
                lam_race=lam_race,
                s_scale_gender=s_scale_gender,
                s_scale_race=s_scale_race,
                neutral=neutral if neutral is not None else False,
                threshold=threshold,
                device=device
            )
            
            # Combine with existing logits processors
            if 'logits_processor' not in kwargs:
                kwargs['logits_processor'] = LogitsProcessorList()
            kwargs['logits_processor'].append(logit_processor)
            
        elif is_race_mode:
            # Multi-class race detection using hard-coded tokens
            from transformers.generation.logits_process import LogitsProcessorList
            
            race_probs = kwargs.pop('race_probs')
            race_map = kwargs.pop('race_map', None)
            
            logit_processor = MultiClassRaceLogitProcessor(
                race_probs=race_probs,
                tokenizer=vqa_tokenizer,
                lam=lam if lam is not None else 0.9,
                device=device,
                race_map=race_map
            )
            
            # Combine with existing logits processors
        elif is_gender_mode:
            # Binary gender detection using text classifier and token bias
            from transformers.generation.logits_process import LogitsProcessorList
            
            logit_processor = LogitAdjustmentProcessor(
                token_bias=token_bias,
                text_classifier=text_classifier,
                tokenizer=vqa_tokenizer,
                alpha_tokenizer=alpha_tokenizer,
                lam=lam if lam is not None else 0.9,
                neutral=neutral if neutral is not None else False,
                s_scale=s_scale if s_scale is not None else 0.0,
                threshold=threshold,
                device=device
            )
            
            # Combine with existing logits processors
        else:
            logit_processor = None
        
        if logit_processor is not None:
            if logits_processor is None:
                logits_processor = LogitsProcessorList([logit_processor])
            else:
                if isinstance(logits_processor, LogitsProcessorList):
                    processors = list(logits_processor)
                    processors.append(logit_processor)
                    logits_processor = LogitsProcessorList(processors)
                elif isinstance(logits_processor, list):
                    processors = list(logits_processor)
                    processors.append(logit_processor)
                    logits_processor = LogitsProcessorList(processors)
                else:
                    logits_processor = LogitsProcessorList([logits_processor, logit_processor])
        
        # Set default generation parameters if not provided
        # Add repetition_penalty to prevent repetitive token generation
        if 'repetition_penalty' not in kwargs:
            kwargs['repetition_penalty'] = 1.1
        
        # Call parent generate method
        return super(type(self), self).generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs
        )
    
    def generate_with_debiasing(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        mode: str = 'naive',
        **kwargs
    ):
        """
        Wrapper for generate with debiasing modes.
        """
        return self.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            mode=mode,
            **kwargs
        )

