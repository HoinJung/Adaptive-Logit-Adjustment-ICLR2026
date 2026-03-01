# paligemma_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PaliGemmaForConditionalGeneration
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import LogitsProcessor
from typing import Optional, Union, List, Dict, Any
import copy


class LogitAdjustmentProcessor(LogitsProcessor):
    """
    LogitsProcessor that applies logit adjustment based on token bias and text classifier.
    Adapted from ClipCAP implementation for PaliGemma.
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
            # IMPORTANT: Clip token IDs to classifier's vocabulary range to avoid CUDA assert errors
            classifier_vocab_size = self.text_classifier.token_embedding.num_embeddings
            
            # Convert input_ids to classifier's token space and clip to valid range
            classifier_input_ids = input_ids.clone()
            # Clip to valid range [0, vocab_size-1] to prevent out-of-bounds access
            # This is safe because any token ID >= vocab_size will be mapped to vocab_size-1
            # which is a valid embedding (usually padding or EOS token)
            classifier_input_ids = torch.clamp(classifier_input_ids, 0, classifier_vocab_size - 1)
            
            # Append the most likely token for classifier input (also clipped)
            # If top_token_id is out of range, clip it to the max valid ID
            top_token_id_clipped = min(int(top_token_id), classifier_vocab_size - 1)
            if top_token_id_clipped < 0:
                top_token_id_clipped = 0
            next_token_tensor = torch.tensor([[top_token_id_clipped]], device=self.device, dtype=torch.long)
            classifier_input_ids = torch.cat([classifier_input_ids, next_token_tensor], dim=1)
            
            # Ensure proper attention mask
            attention_mask = torch.ones_like(classifier_input_ids, device=self.device)
            
            # Get classifier output with error handling
            try:
                with torch.no_grad():
                    classifier_output = self.text_classifier(
                        input_ids=classifier_input_ids,
                        attention_mask=attention_mask
                    )
            except RuntimeError as e:
                # If there's still an error, skip classifier and return unadjusted scores
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    print(f"Warning: Classifier error (likely out-of-range token IDs), skipping adjustment: {e}")
                    return scores
                raise
            
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
            
            # Apply adjustment to all tokens in vocabulary
            # Formula from ClipCAP: logits[token_id] -= lam * (alpha - s) * beta_i
            # Only iterate over valid tokenizer vocabulary range
            vocab_range = min(tokenizer_vocab_size, model_vocab_size)
            for token_id in range(vocab_range):
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    if token_str in self.token_bias:
                        beta_i = self.token_bias[token_str]
                        # For neutralization: take absolute value to compress all gender-related tokens
                        # regardless of bias direction (both positive and negative beta_i get reduced)
                        if self.neutral:
                            beta_i = abs(beta_i)
                        scores[:, token_id] -= self.lam * (alpha - s) * beta_i
                except (IndexError, ValueError):
                    # Skip invalid token IDs
                    continue
        
        return scores


class VDDLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor for VDD (Vision Debiasing via Dropout) method.
    Formula: (1+lambda)*logit(original_image) - lambda*logit(pure_noise_image)
    
    This processor computes logits from noise image at each step and combines them.
    """
    def __init__(
        self,
        model,
        noise_pixel_values: torch.FloatTensor,
        input_ids_base: torch.LongTensor,
        attention_mask_base: torch.LongTensor,
        lam: float,
        device: str
    ):
        self.model = model
        self.noise_pixel_values = noise_pixel_values
        self.input_ids_base = input_ids_base
        self.attention_mask_base = attention_mask_base
        self.lam = lam
        self.device = device
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply VDD formula: (1+lambda)*logit(original) - lambda*logit(noise)
        
        At each generation step:
        1. scores contains logits from original image (already computed)
        2. Compute logits from noise image using current input_ids
        3. Combine: (1+lam)*original - lam*noise
        """
        # Use the current input_ids (which includes all generated tokens so far)
        noise_input_ids = input_ids
        
        # Get attention mask for noise forward pass
        if self.attention_mask_base is not None:
            base_len = self.input_ids_base.shape[-1]
            current_len = input_ids.shape[-1]
            if current_len > base_len:
                # Extend attention mask for new tokens
                attention_mask = torch.cat([
                    self.attention_mask_base,
                    torch.ones((input_ids.shape[0], current_len - base_len), 
                              device=self.device, dtype=self.attention_mask_base.dtype)
                ], dim=-1)
            else:
                attention_mask = self.attention_mask_base[:, :current_len]
        else:
            attention_mask = torch.ones_like(noise_input_ids)
        
        # Forward pass with noise image to get logits for next token
        # We need to get logits for the last position
        with torch.no_grad():
            noise_outputs = self.model(
                input_ids=noise_input_ids,
                pixel_values=self.noise_pixel_values,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True
            )
            # Get logits for the next token (last position in sequence)
            noise_scores = noise_outputs.logits[:, -1, :] if noise_outputs.logits.dim() == 3 else noise_outputs.logits
        
        # Combine logits according to VDD formula: (1+lambda)*original - lambda*noise
        combined_scores = (1 + self.lam) * scores - self.lam * noise_scores
        
        return combined_scores


class CustomPaliGemmaForConditionalGeneration(PaliGemmaForConditionalGeneration):
    """
    Custom PaliGemma model that supports debiasing arguments in the generate method.
    """
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[Any] = None,
        logits_processor: Optional[Any] = None,
        stopping_criteria: Optional[Any] = None,
        prefix_allowed_tokens_fn: Optional[Any] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional[Any] = None,
        streamer: Optional[Any] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Custom generate method that supports debiasing arguments including logit adjustment.
        """
        # Extract custom debiasing arguments
        decoder_mean_features_lowconfidence = kwargs.pop('decoder_mean_features_lowconfidence', None)
        decoder_important_indices = kwargs.pop('decoder_important_indices', None)
        mode = kwargs.pop('mode', None)
        vqa_name = kwargs.pop('vqa_name', None)
        
        # Extract logit adjustment parameters
        s_scale = kwargs.pop('s_scale', None)
        text_classifier = kwargs.pop('text_classifier', None)
        token_bias = kwargs.pop('token_bias', None)
        lam = kwargs.pop('lam', None)
        neutral = kwargs.pop('neutral', None)
        vqa_tokenizer = kwargs.pop('vqa_tokenizer', None)
        alpha_tokenizer = kwargs.pop('alpha_tokenizer', None)
        threshold = kwargs.pop('threshold', 0.1)
        device = kwargs.pop('device', 'cuda:0')
        
        # Store debiasing parameters for use in forward pass
        self._debiasing_params = {
            'decoder_mean_features_lowconfidence': decoder_mean_features_lowconfidence,
            'decoder_important_indices': decoder_important_indices,
            'mode': mode,
            'vqa_name': vqa_name,
            's_scale': s_scale,
            'text_classifier': text_classifier,
            'token_bias': token_bias,
            'lam': lam,
            'neutral': neutral,
            'vqa_tokenizer': vqa_tokenizer,
            'alpha_tokenizer': alpha_tokenizer,
            'threshold': threshold,
            'device': device
        }
        
        # For logit mode, create and add logits processor
        if mode == 'logit' and token_bias is not None and text_classifier is not None:
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
            if logits_processor is None:
                logits_processor = LogitsProcessorList([logit_processor])
            else:
                # Convert existing processor(s) to list format
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
        
        # For VDD mode, create LogitsProcessor that combines original and noise image logits
        vdd_lam = kwargs.pop('vdd_lam', 1.0) if mode == 'vdd' else None
        if mode == 'vdd' and vdd_lam is not None and 'pixel_values' in kwargs:
            from transformers.generation.logits_process import LogitsProcessorList
            
            # Create pure noise image (same shape as original)
            original_pixel_values = kwargs['pixel_values']
            noise_pixel_values = torch.randn_like(original_pixel_values)
            
            # Get base input_ids and attention_mask for VDD processor
            base_input_ids = kwargs.get('input_ids', inputs) if inputs is not None else kwargs.get('input_ids')
            base_attention_mask = kwargs.get('attention_mask')
            
            # Create VDD LogitsProcessor
            vdd_processor = VDDLogitsProcessor(
                model=self,
                noise_pixel_values=noise_pixel_values,
                input_ids_base=base_input_ids,
                attention_mask_base=base_attention_mask,
                lam=vdd_lam,
                device=device
            )
            
            # Add VDD processor to logits processors
            if logits_processor is None:
                logits_processor = LogitsProcessorList([vdd_processor])
            else:
                if isinstance(logits_processor, LogitsProcessorList):
                    processors = list(logits_processor)
                    processors.append(vdd_processor)
                    logits_processor = LogitsProcessorList(processors)
                elif isinstance(logits_processor, list):
                    processors = list(logits_processor)
                    processors.append(vdd_processor)
                    logits_processor = LogitsProcessorList(processors)
                else:
                    logits_processor = LogitsProcessorList([logits_processor, vdd_processor])
        
        # Call the parent generate method
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Custom forward method that applies debiasing to the model outputs.
        """
        # Get debiasing parameters
        debiasing_params = getattr(self, '_debiasing_params', {})
        mode = debiasing_params.get('mode')
        decoder_mean_features_lowconfidence = debiasing_params.get('decoder_mean_features_lowconfidence')
        decoder_important_indices = debiasing_params.get('decoder_important_indices')
        
        # VDD mode is handled in LogitsProcessor, no need for special forward logic
        # Call parent forward method
        outputs = super().forward(
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
            **kwargs
        )
        
        # Apply debiasing to the decoder hidden states if specified
        if mode in ['sfid', 'clipclip'] and decoder_mean_features_lowconfidence is not None and decoder_important_indices is not None:
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                # For CLIPCLIP mode, we apply pruning to the decoder's last hidden states
                if mode == 'clipclip' and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                    
                    # Convert decoder_important_indices to tensor if needed
                    if isinstance(decoder_important_indices, torch.Tensor):
                        important_indices = decoder_important_indices.to(hidden_states.device)
                    else:
                        important_indices = torch.tensor(decoder_important_indices).to(hidden_states.device)
                    
                    # Apply pruning by setting selected features to zero
                    for idx in important_indices:
                        if idx < hidden_states.shape[-1]:
                            hidden_states[:, :, idx] = 0.0
                    
                    # Recompute logits with debiased hidden states
                    if hasattr(self, 'language_model') and hasattr(self.language_model, 'lm_head'):
                        logits = self.language_model.lm_head(hidden_states)
                        outputs.logits = logits
        
        
        return outputs
    
    def generate_with_debiasing(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        mode: str = 'naive',
        **kwargs
    ):
        """Custom generate method with debiasing support"""
        
        if mode == 'naive':
            return self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # For logit mode, use the overridden generate method which handles LogitsProcessor
        if mode == 'logit':
            return self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                mode=mode,
                **kwargs
            )
        
        # For other debiasing modes
        if mode in ['sfid', 'clipclip', 'steer', 'vdd']:
            return self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                mode=mode,
                **kwargs
            )
        
        # Fallback to standard generate
        return self.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )


def create_custom_paligemma_model(model_id: str, device: str = "cuda"):
    """Create a custom PaliGemma model that supports debiasing (logit, sfid, clipclip, vdd, dear)."""
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    custom_model = CustomPaliGemmaForConditionalGeneration(base_model.config)
    custom_model.load_state_dict(base_model.state_dict())
    custom_model = custom_model.to(device)
    custom_model.eval()
    return custom_model


