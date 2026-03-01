# llava_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation.logits_process import LogitsProcessor
from torch.nn import CrossEntropyLoss
import copy

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


class IntersectionLogitProcessor(LogitsProcessor):
    """
    LogitsProcessor for intersectional debiasing (both gender and race).
    Applies adjustments using both gender and race classifiers and token biases.
    """
    def __init__(
        self,
        gender_token_bias: dict,
        race_token_bias: dict,
        gender_text_classifier,
        race_text_classifier,
        tokenizer,
        alpha_tokenizer,
        lam: float = 0.9,
        lam_gender: float = None,
        lam_race: float = None,
        s_scale_gender: float = None,
        s_scale_race: float = None,
        neutral: bool = False,
        threshold: float = 0.1,
        device: str = 'cuda:0'
    ):
        self.gender_token_bias = gender_token_bias
        self.race_token_bias = race_token_bias
        self.gender_text_classifier = gender_text_classifier
        self.race_text_classifier = race_text_classifier
        self.tokenizer = tokenizer
        self.alpha_tokenizer = alpha_tokenizer
        self.lam = lam
        # Use separate lambdas if provided, otherwise use the same lam for both
        self.lam_gender = lam_gender if lam_gender is not None else lam
        self.lam_race = lam_race if lam_race is not None else lam
        self.s_scale_gender = s_scale_gender
        self.s_scale_race = s_scale_race
        self.neutral = neutral
        self.threshold = threshold
        self.device = device
        
        # Pre-compute token_id -> beta_value mappings for fast lookup
        # This avoids iterating through entire vocabulary at every generation step
        self.token_id_to_beta_gender = {}
        self.token_id_to_beta_race = {}
        self.token_ids_with_bias = []
        if (gender_token_bias or race_token_bias) and tokenizer is not None:
            print("Pre-computing token_id to beta mapping for intersection logit adjustment...")
            tokenizer_vocab_size = len(tokenizer)
            for token_id in range(tokenizer_vocab_size):
                try:
                    token_str = tokenizer.convert_ids_to_tokens(token_id)
                    has_bias = False
                    if token_str in gender_token_bias:
                        beta_value = gender_token_bias[token_str]
                        if abs(beta_value) >= threshold:
                            self.token_id_to_beta_gender[token_id] = beta_value
                            has_bias = True
                    if token_str in race_token_bias:
                        beta_value = race_token_bias[token_str]
                        if abs(beta_value) >= threshold:
                            self.token_id_to_beta_race[token_id] = beta_value
                            has_bias = True
                    if has_bias:
                        self.token_ids_with_bias.append(token_id)
                except (IndexError, ValueError):
                    continue
            print(f"Pre-computed {len(self.token_ids_with_bias)} tokens with bias >= {threshold} (out of {tokenizer_vocab_size} total)")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Adjust logits using both gender and race biases.
        
        Args:
            input_ids: Current input sequence
            scores: Logits for next token prediction [batch_size, vocab_size]
        
        Returns:
            Adjusted logits
        """
        # Get tokenizer vocabulary size
        tokenizer_vocab_size = len(self.tokenizer)
        model_vocab_size = scores.shape[-1]
        
        # Find the most likely token
        top_token_id = torch.argmax(scores, dim=-1).item()
        
        # Check if token_id is valid for tokenizer
        if top_token_id >= tokenizer_vocab_size:
            return scores
        
        try:
            token_str = self.tokenizer.convert_ids_to_tokens(top_token_id)
        except (IndexError, ValueError):
            return scores
        
        # Get beta values for both gender and race (from top token, for threshold check)
        beta_gender = 0.0
        beta_race = 0.0
        if token_str in self.gender_token_bias:
            beta_gender = self.gender_token_bias[token_str]
        if token_str in self.race_token_bias:
            beta_race = self.race_token_bias[token_str]
        
        # Check if we need to run each classifier
        # If lambda is non-zero, we should run the classifier to compute alpha for adjustments
        # The threshold check on top token is an optimization, but if lambda is explicitly set,
        # we should run the classifier (threshold check can still be used as a quick filter)
        # However, to match race-only mode behavior, we'll use the same threshold logic:
        # run if top token has bias above threshold (this matches LogitAdjustmentProcessor behavior)
        # BUT: if lambda is non-zero and we have tokens in the bias dict, we should run it
        need_gender_classifier = (self.lam_gender != 0) and (abs(beta_gender) >= self.threshold)
        need_race_classifier = (self.lam_race != 0) and (abs(beta_race) >= self.threshold)
        
        # Optimization: If token_bias is hardcoded (only 0.0 and 1.0 values), skip classifier entirely!
        # Fast check: sample a few values to see if it's hardcoded (avoid checking all values)
        gender_is_hardcoded = False
        if self.gender_token_bias:
            sample_values = list(self.gender_token_bias.values())[:100]  # Sample first 100
            gender_is_hardcoded = all(v in (0.0, 1.0) for v in sample_values)
        
        race_is_hardcoded = False
        if self.race_token_bias:
            sample_values = list(self.race_token_bias.values())[:100]  # Sample first 100
            race_is_hardcoded = all(v in (0.0, 1.0) for v in sample_values)
        
        # KEY OPTIMIZATION: If both are hardcoded, skip classifier entirely!
        # With hardcoded bias, we don't need alpha from classifier - we can use fixed values
        if gender_is_hardcoded and race_is_hardcoded:
            # Both hardcoded: skip classifier completely, use fixed alpha values
            skip_classifier = True
            # For hardcoded case, we can use alpha = 1.0 (or 0.0) directly without classifier
            # Since bias is 0.0 or 1.0, and we want to debias, we can use alpha = 1.0 for all tokens
            # This means: adjustment = lam * (1.0 - s) * beta
            # For neutral mode: s_gender = 0, s_race = -1
            # For non-neutral: s_gender = s_scale_gender, s_race = -1
        elif gender_is_hardcoded:
            # Only gender hardcoded: skip gender classifier
            need_gender_classifier = False
            skip_classifier = not need_race_classifier
        elif race_is_hardcoded:
            # Only race hardcoded: skip race classifier  
            need_race_classifier = False
            skip_classifier = not need_gender_classifier
        else:
            # Not hardcoded: use original logic
            if self.lam_gender != 0 and not need_gender_classifier:
                if self.gender_token_bias:
                    need_gender_classifier = True
            
            if self.lam_race != 0 and not need_race_classifier:
                if self.race_token_bias:
                    need_race_classifier = True
            
            alpha_gender = 0.0
        alpha_race = 0.0
        
        # If both are hardcoded, use fixed alpha values (no classifier needed at all!)
        if gender_is_hardcoded and race_is_hardcoded:
            # For hardcoded bias, we can use a fixed alpha value
            # Using alpha = 1.0 means we apply full debiasing adjustment
            # This completely skips classifier calls - huge speedup!
            alpha_gender = 1.0
            alpha_race = 1.0
            skip_classifier = True  # Ensure we skip classifier
        elif gender_is_hardcoded:
            # Only gender hardcoded: use fixed alpha for gender, run race classifier if needed
            alpha_gender = 1.0
            if not skip_classifier and need_race_classifier:
                # Only run race classifier
                skip_classifier = False
        elif race_is_hardcoded:
            # Only race hardcoded: use fixed alpha for race, run gender classifier if needed
            alpha_race = 1.0
            if not skip_classifier and need_gender_classifier:
                # Only run gender classifier
                skip_classifier = False
        
        if not skip_classifier:
            # Prepare sequence for classifiers
            classifier_input_ids = input_ids.clone()
            next_token_tensor = torch.tensor([[top_token_id]], device=self.device)
            classifier_input_ids = torch.cat([classifier_input_ids, next_token_tensor], dim=1)
            
            # Clamp token IDs - use the classifier that's actually needed
            # If both are needed, use the minimum vocab size to be safe
            vocab_size = None
            if need_gender_classifier and hasattr(self.gender_text_classifier, 'token_embedding'):
                vocab_size = self.gender_text_classifier.token_embedding.num_embeddings
            if need_race_classifier and hasattr(self.race_text_classifier, 'token_embedding'):
                race_vocab_size = self.race_text_classifier.token_embedding.num_embeddings
                if vocab_size is None:
                    vocab_size = race_vocab_size
                else:
                    # Use minimum to ensure compatibility with both
                    vocab_size = min(vocab_size, race_vocab_size)
            if vocab_size is not None:
                classifier_input_ids = torch.clamp(classifier_input_ids, 0, vocab_size - 1)
            
            attention_mask = torch.ones_like(classifier_input_ids, device=self.device, dtype=torch.long)
            
            # Get gender classifier output (only if needed)
            if need_gender_classifier:
                with torch.no_grad():
                    gender_output = self.gender_text_classifier(
                        input_ids=classifier_input_ids,
                        attention_mask=attention_mask
                    )
                    alpha_gender = gender_output["logits"].squeeze()
                    if alpha_gender.dim() > 0:
                        alpha_gender = alpha_gender[1] if len(alpha_gender) > 1 else alpha_gender[0]
                    alpha_gender = 2 * torch.sigmoid(alpha_gender) - 1
            
            # Get race classifier output (only if needed)
            if need_race_classifier:
                with torch.no_grad():
                    race_output = self.race_text_classifier(
                        input_ids=classifier_input_ids,
                        attention_mask=attention_mask
                    )
                    alpha_race = race_output["logits"].squeeze()
                    if alpha_race.dim() > 0:
                        alpha_race = alpha_race[1] if len(alpha_race) > 1 else alpha_race[0]
                    alpha_race = 2 * torch.sigmoid(alpha_race) - 1
        
        # Determine s values
        if self.neutral:
            s_gender = 0
            alpha_gender = abs(alpha_gender)
            s_race = -1
        else:
            s_gender = self.s_scale_gender if self.s_scale_gender is not None else 0
            s_race = -1
        
        # Apply adjustments to tokens with bias (optimized: only iterate over pre-computed tokens)
        # Formula: logits[token_id] -= lam * [(alpha_gender - s_gender) * beta_gender + (alpha_race - s_race) * beta_race]
        adjustment_gender = self.lam_gender * (alpha_gender - s_gender)
        adjustment_race = self.lam_race * (alpha_race - s_race)
        for token_id in self.token_ids_with_bias:
            if token_id < model_vocab_size:  # Ensure token_id is within model vocab range
                adjustment = 0.0
                
                if token_id in self.token_id_to_beta_gender:
                    beta_i_gender = self.token_id_to_beta_gender[token_id]
                    if self.neutral:
                        beta_i_gender = abs(beta_i_gender)
                    adjustment += adjustment_gender * beta_i_gender
                
                if token_id in self.token_id_to_beta_race:
                    beta_i_race = self.token_id_to_beta_race[token_id]
                    if self.neutral:
                        beta_i_race = abs(beta_i_race)
                    adjustment += adjustment_race * beta_i_race
                
                if adjustment != 0.0:
                    scores[:, token_id] -= adjustment
        
        return scores


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
        generated_sequence: Optional[List[int]] = None,
        use_occupation_bias: bool = False,
        debug: bool = False
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
        self.use_occupation_bias = use_occupation_bias
        self.debug = debug
        self.debug_info = []  # Store debug information
        self._debug_first_call = True  # Track if this is the first call
        
        # Pre-compute token_id -> beta_value mappings for fast lookup
        # This avoids iterating through entire vocabulary at every generation step
        self.token_id_to_beta = {}
        self.token_ids_with_bias = []
        if token_bias and tokenizer is not None:
            print("Pre-computing token_id to beta mapping for logit adjustment...")
            tokenizer_vocab_size = len(tokenizer)
            for token_id in range(tokenizer_vocab_size):
                try:
                    token_str = tokenizer.convert_ids_to_tokens(token_id)
                    if token_str in token_bias:
                        beta_value = token_bias[token_str]
                        if abs(beta_value) >= threshold:
                            self.token_id_to_beta[token_id] = beta_value
                            self.token_ids_with_bias.append(token_id)
                except (IndexError, ValueError):
                    continue
            print(f"Pre-computed {len(self.token_ids_with_bias)} tokens with bias >= {threshold} (out of {tokenizer_vocab_size} total)")
        
        # Hard-coded beta values for occupation bias mode
        if self.use_occupation_bias:
            # Doctor-related tokens: beta = -0.5
            # Nurse-related tokens: beta = -0.5
            # Note: With formula z = z - lam*(alpha+s)*beta:
            #   - Female image (s<0) + nurse word (alpha>0, full context): alpha+s ≈ 0.17
            #   - beta_nurse = -0.5 → -lam*(0.17)*(-0.5) = +0.085*lam → nurse logit 감소 ✓
            #   - Female image (s<0) + doctor word (alpha<0 or small): alpha+s < 0
            #   - beta_doctor = -0.5 → -lam*(음수)*(-0.5) = 음수 → z -= 음수 = z += 양수 → doctor logit 증가 ✓
            #   - Female image (s<0) + nurse word (alpha<0) → alpha+s < 0
            #   - beta_nurse > 0 → lam*(음수)*(양수) = 음수 → nurse logit 감소 ✓
            #   - Female image (s<0) + doctor word (alpha>0) → alpha+s could be small
            #   - beta_doctor < 0 → lam*(작은값)*(음수) = 작은양수 → doctor logit 증가 ✓
            # All other tokens: beta = 0 (no adjustment)
            self.occupation_beta_map = {}
            doctor_keywords = ['doctor', 'physician', 'surgeon', 'medic']
            nurse_keywords = ['nurse', 'nur']
            
            # Build token mapping
            if tokenizer is not None:
                vocab_size = len(tokenizer)
                doctor_tokens = []
                nurse_tokens = []
                for token_id in range(vocab_size):
                    try:
                        token_str = tokenizer.convert_ids_to_tokens(token_id)
                        token_lower = token_str.lower().replace('▁', '').replace('Ġ', '')
                        
                        # Check if token matches doctor or nurse
                        is_doctor = any(kw in token_lower for kw in doctor_keywords)
                        is_nurse = any(kw in token_lower for kw in nurse_keywords)
                        
                        if is_doctor:
                            self.occupation_beta_map[token_str] = 0.5 
                            doctor_tokens.append(token_str)
                        elif is_nurse:
                            self.occupation_beta_map[token_str] = -0.5 
                            nurse_tokens.append(token_str)
                    except:
                        continue
                
                # Debug: print found tokens (only first time)
                if not hasattr(LogitAdjustmentProcessor, '_debug_printed'):
                    print(f"\n[Occupation Bias] Found {len(doctor_tokens)} doctor-related tokens: {doctor_tokens[:10]}")
                    print(f"[Occupation Bias] Found {len(nurse_tokens)} nurse-related tokens: {nurse_tokens[:10]}")
                    print(f"[Occupation Bias] Total non-zero beta tokens: {len(self.occupation_beta_map)}")
                    print(f"[Occupation Bias] All other tokens have beta = 0 (no adjustment)")
                    LogitAdjustmentProcessor._debug_printed = True
    
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
        # For occupation_bias mode, check occupation_beta_map instead of token_bias
        if self.use_occupation_bias:
            beta_value = 0.0
            if token_str in self.occupation_beta_map:
                beta_value = self.occupation_beta_map[token_str]
            # For occupation bias mode, we need to run classifier to get alpha
            # even if top token is not an occupation token, because we need to adjust
            # occupation tokens in the vocabulary. So we check if occupation_beta_map
            # has any tokens (it should, since we built it in __init__)
            # We always run classifier in occupation_bias mode to ensure alpha is computed
            skip_classifier = False  # Always run classifier in occupation_bias mode
        else:
            beta_value = 0.0
            if token_str in self.token_bias:
                beta_value = self.token_bias[token_str]
            # Check if we should run classifier (threshold check)
            # Similar to ClipCAP: skip_classifier = (abs(beta_value) < threshold)
            skip_classifier = (abs(beta_value) < self.threshold)
        
        if not skip_classifier:
            # Prepare sequence for classifier
            # Use only the most likely next token (without question context)
            # This ensures alpha reflects the token's inherent gender association
            next_token_tensor = torch.tensor([[top_token_id]], device=self.device)
            classifier_input_ids = next_token_tensor
            
            # Clamp token IDs to valid vocabulary range (classifier will handle this, but do it here for safety)
            if hasattr(self.text_classifier, 'token_embedding'):
                vocab_size = self.text_classifier.token_embedding.num_embeddings
                classifier_input_ids = torch.clamp(classifier_input_ids, 0, vocab_size - 1)
            
            # Ensure proper attention mask
            attention_mask = torch.ones_like(classifier_input_ids, device=self.device, dtype=torch.long)
            
            # Get classifier output (token only, no question)
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
            
            # Debug: Store alpha and s values (only for first token generation step)
            if self.debug and self._debug_first_call:
                alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                s_val = s if isinstance(s, (int, float)) else s.item() if isinstance(s, torch.Tensor) else s
                top_token_str = self.tokenizer.convert_ids_to_tokens(top_token_id)
                
                debug_entry = {
                    'step': 0,
                    'top_token': top_token_str,
                    'top_token_id': top_token_id,
                    'alpha': alpha_val,
                    's': s_val,
                    'alpha_minus_s': alpha_val - s_val,  # Use original ClipCAP formula
                    'lam': self.lam,
                    'neutral': self.neutral
                }
                self.debug_info.append(debug_entry)
                
                print(f"[DEBUG] Step 0 (first generation step): top_token='{top_token_str}' (id={top_token_id})")
                print(f"  alpha (token only, no question): {alpha_val:.4f}")
                if self.use_occupation_bias:
                    print(f"  s: {s_val:.4f} (from profession classifier: positive=doctor, negative=nurse), alpha-s: {alpha_val - s_val:.4f}, lam: {self.lam}, neutral: {self.neutral}")
                else:
                    print(f"  s: {s_val:.4f}, alpha-s: {alpha_val - s_val:.4f}, lam: {self.lam}, neutral: {self.neutral}")
                self._debug_first_call = False  # Mark that we've printed the first call
            
            # Apply adjustment to all tokens in vocabulary
            # Only iterate over valid tokenizer vocabulary range
            vocab_range = min(tokenizer_vocab_size, model_vocab_size)
            
            if self.use_occupation_bias:
                # Occupation bias mode: use original ClipCAP formula (alpha - s) with hard-coded beta values
                # Formula: logits[token_id] -= lam * (alpha - s) * beta_i
                # Note: alpha is computed from top_token only (no question context)
                # - doctor tokens should have positive alpha
                # - nurse tokens should have negative alpha
                # - doctor images: s > 0 (from profession classifier)
                # - nurse images: s < 0 (from profession classifier)
                # The discrepancy (alpha - s) is large when there's a mismatch:
                #   - doctor image (s>0) + nurse token (α<0): (α-s) << 0 → reduces nurse logits
                #   - nurse image (s<0) + doctor token (α>0): (α-s) >> 0 → reduces doctor logits
                # Only doctor/nurse tokens have non-zero beta, all others are 0
                
                # Debug: Collect all doctor/nurse token alphas and adjustments for first step
                debug_adjustments = []
                debug_token_alphas = []
                
                # For each doctor/nurse token, compute its alpha value
                if self.debug and self._debug_first_call:
                    for token_str, beta_i in self.occupation_beta_map.items():
                        try:
                            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                            if token_id is None or token_id >= vocab_range:
                                continue
                            
                            # Compute alpha for this token (token only, no question)
                            token_tensor = torch.tensor([[token_id]], device=self.device)
                            if hasattr(self.text_classifier, 'token_embedding'):
                                vocab_size = self.text_classifier.token_embedding.num_embeddings
                                token_tensor = torch.clamp(token_tensor, 0, vocab_size - 1)
                            
                            token_attention_mask = torch.ones_like(token_tensor, device=self.device, dtype=torch.long)
                            
                            with torch.no_grad():
                                token_classifier_output = self.text_classifier(
                                    input_ids=token_tensor,
                                    attention_mask=token_attention_mask
                                )
                            
                            token_alpha_raw = token_classifier_output["logits"].squeeze()
                            
                            if self.neutral:
                                if token_alpha_raw.dim() > 0:
                                    token_alpha_raw = token_alpha_raw[1] if len(token_alpha_raw) > 1 else token_alpha_raw[0]
                                token_alpha = 2 * torch.sigmoid(token_alpha_raw) - 1
                                token_alpha = abs(token_alpha)
                            else:
                                if token_alpha_raw.dim() > 0:
                                    token_alpha_raw = token_alpha_raw[1] if len(token_alpha_raw) > 1 else token_alpha_raw[0]
                                token_alpha = 2 * torch.sigmoid(token_alpha_raw) - 1
                            
                            token_alpha_val = token_alpha.item() if isinstance(token_alpha, torch.Tensor) else token_alpha
                            debug_token_alphas.append({
                                'token': token_str,
                                'token_id': token_id,
                                'alpha': token_alpha_val
                            })
                        except (IndexError, ValueError, KeyError):
                            continue
                
                for token_id in range(vocab_range):
                    try:
                        token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                        # Only apply adjustment if token is in occupation_beta_map
                        # All other tokens have beta = 0 (no adjustment)
                        if token_str in self.occupation_beta_map:
                            beta_i = self.occupation_beta_map[token_str]
                            # Use original ClipCAP formula: (alpha - s) for occupation bias
                            # Note: We use the alpha from top_token, not from this token
                            # This is the current implementation - alpha is computed once from top_token
                            adjustment = self.lam * (alpha - s) * beta_i
                            scores[:, token_id] -= adjustment
                            
                            # Debug: Collect adjustment info for doctor/nurse tokens (only for first step)
                            if self.debug and self._debug_first_call:
                                alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                                s_val = s if isinstance(s, (int, float)) else s.item() if isinstance(s, torch.Tensor) else s
                                adj_val = adjustment.item() if isinstance(adjustment, torch.Tensor) else adjustment
                                debug_adjustments.append({
                                    'token': token_str,
                                    'token_id': token_id,
                                    'beta': beta_i,
                                    'adjustment': adj_val
                                })
                        # else: beta_i = 0, so no adjustment (implicit)
                    except (IndexError, ValueError):
                        # Skip invalid token IDs
                        continue
                
                # Debug: Print all doctor/nurse token alphas and adjustments (only for first step)
                if self.debug and self._debug_first_call:
                    alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
                    s_val = s if isinstance(s, (int, float)) else s.item() if isinstance(s, torch.Tensor) else s
                    
                    if debug_token_alphas:
                        print(f"[DEBUG] Occupation token alphas (computed from each token, no question):")
                        for token_alpha_info in debug_token_alphas:
                            print(f"  Token '{token_alpha_info['token']}' (id={token_alpha_info['token_id']}): alpha={token_alpha_info['alpha']:.4f}")
                    
                    if debug_adjustments:
                        print(f"[DEBUG] Occupation token adjustments (using top_token alpha={alpha_val:.4f}, s={s_val:.4f}, alpha-s={alpha_val - s_val:.4f}):")
                        for adj_info in debug_adjustments:
                            print(f"  Token '{adj_info['token']}' (id={adj_info['token_id']}): beta={adj_info['beta']:.4f}, adjustment={adj_info['adjustment']:.4f}")
            else:
                # Standard mode: use alpha - s formula (original ClipCAP formula)
                # Formula: logits[token_id] -= lam * (alpha - s) * beta_i
                # Optimized: only iterate over pre-computed tokens with bias
                adjustment_value = self.lam * (alpha - s)
                for token_id in self.token_ids_with_bias:
                    if token_id < model_vocab_size:  # Ensure token_id is within model vocab range
                        beta_i = self.token_id_to_beta[token_id]
                        if self.neutral:
                            beta_i = abs(beta_i)
                        # Original formula: z = z - lam * (alpha - s) * beta
                        scores[:, token_id] -= adjustment_value * beta_i
        
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

class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.interim_hidden_state = None
        
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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Apply DEAR adaptor if provided
        if dear_adaptor is not None and pixel_values is not None:
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                vision_features = vision_outputs.pooler_output
                adapted_features = dear_adaptor(vision_features)
                # Replace the vision features in the model
                # This is a simplified approach - in practice, you might need to modify the vision tower output
        
        # Call parent forward method
        # Fix num_logits_to_keep if it's None (transformers library issue)
        if 'num_logits_to_keep' in kwargs and kwargs['num_logits_to_keep'] is None:
            kwargs['num_logits_to_keep'] = 0
        
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
        
        # Apply debiasing modifications to hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Get last layer hidden states
            
            if mode in ['sfid', 'clipclip'] and text_important_indices is not None and text_mean_features_lowconfidence is not None:
                # Apply SFID/CLIP-CLIP modifications
                hidden_dim_size = hidden_states.shape[1]
                text_mean_features_lowconfidence = text_mean_features_lowconfidence.unsqueeze(0).unsqueeze(1)
                text_mean_features_lowconfidence = text_mean_features_lowconfidence.expand(1, hidden_dim_size, hidden_states.shape[-1])
                hidden_states[:, :, text_important_indices] = text_mean_features_lowconfidence[:, :, text_important_indices]
            
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
        Similar to Paligemma implementation.
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
        use_occupation_bias = kwargs.pop('use_occupation_bias', False)
        debug = kwargs.pop('debug', False)
        
        # For logit mode, create and add logits processor
        # Check if this is intersection mode, race detection (multi-class), or gender detection (binary)
        is_intersection_mode = mode == 'logit' and 'gender_text_classifier' in kwargs and 'race_text_classifier' in kwargs
        is_race_mode = mode == 'logit' and token_bias is None and text_classifier is None and 'race_probs' in kwargs
        is_gender_mode = mode == 'logit' and token_bias is not None and text_classifier is not None and not is_intersection_mode
        
        if is_intersection_mode:
            # Intersection mode: both gender and race debiasing
            from transformers.generation.logits_process import LogitsProcessorList
            
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
            
            # Store logit processor for later use (don't add to kwargs yet to avoid conflicts)
            intersection_logit_processor = logit_processor
            logit_processor = None  # Set to None so it's handled separately below
            
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
                device=device,
                use_occupation_bias=use_occupation_bias,
                debug=debug
            )
            
            # Combine with existing logits processors
        else:
            logit_processor = None
        
        # Handle logits processor for intersection mode
        if is_intersection_mode:
            # For intersection mode, combine with existing logits processor
            existing_logits_processor = kwargs.pop('logits_processor', None)
            if existing_logits_processor is not None:
                if isinstance(existing_logits_processor, LogitsProcessorList):
                    existing_logits_processor.append(intersection_logit_processor)
                    final_logits_processor = existing_logits_processor
                else:
                    final_logits_processor = LogitsProcessorList([existing_logits_processor, intersection_logit_processor])
            else:
                final_logits_processor = LogitsProcessorList([intersection_logit_processor])
        elif logit_processor is not None:
            # Transformers expects LogitsProcessorList, not a single processor
            if logits_processor is None:
                final_logits_processor = LogitsProcessorList([logit_processor])
            else:
                # Convert existing processor(s) to list format
                if isinstance(logits_processor, LogitsProcessorList):
                    # Append to existing list
                    processors = list(logits_processor)
                    processors.append(logit_processor)
                    final_logits_processor = LogitsProcessorList(processors)
                elif isinstance(logits_processor, list):
                    # Append to existing list
                    processors = list(logits_processor)
                    processors.append(logit_processor)
                    final_logits_processor = LogitsProcessorList(processors)
                else:
                    # Single processor, combine with our processor
                    final_logits_processor = LogitsProcessorList([logits_processor, logit_processor])
        else:
            # Use existing logits_processor or None
            final_logits_processor = logits_processor
        
        # Ensure logits_processor is removed from kwargs to avoid duplicate argument error
        kwargs.pop('logits_processor', None)
        
        # Call parent generate method
        return super().generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=final_logits_processor,
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
        """Custom generate method with debiasing support"""
        
        if mode == 'naive':
            return self.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # For debiasing modes, we need to implement custom generation logic
        if mode in ['logit', 'sfid', 'clipclip', 'dear', 'steer', 'vdd']:
            return self._generate_with_debiasing_logic(
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
    
    def _generate_with_debiasing_logic(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        mode: str = 'naive',
        **kwargs
    ):
        """Internal method to handle debiasing generation logic"""
        
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
        
        if mode == 'dear' and 'dear_adaptor' in kwargs:
            # Apply DEAR adaptor to vision features
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                vision_features = vision_outputs.pooler_output
                adapted_features = kwargs['dear_adaptor'](vision_features)
                # Note: This is a simplified approach - you'd need to properly integrate
                # the adapted features into the model's forward pass
        
        # For other modes, use standard generation
        return self.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **{k: v for k, v in kwargs.items() if k not in ['mode', 'dear_adaptor']}
        )
