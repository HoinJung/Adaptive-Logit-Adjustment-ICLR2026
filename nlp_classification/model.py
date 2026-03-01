import torch.nn as nn
import torch.nn.functional as F
import torch

# =======================================================
# 1. Define a simple Transformer classifier model.
# =======================================================
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=2, num_heads=4, max_length=128, num_classes=2, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embeddings
        # print('vocab_size',vocab_size)
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embeddings (for positions 0...max_length-1)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Ensure that inputs longer than max_length are truncated
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        batch_size, seq_len = input_ids.size()
        # Clamp seq_len to max_length to prevent position embedding index errors
        seq_len = min(seq_len, self.max_length)
        # Truncate input_ids and attention_mask if still too long (safety check)
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        
        # Clamp input_ids to valid vocabulary range (0 to vocab_size-1)
        vocab_size = self.token_embedding.num_embeddings
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Ensure positions don't exceed max_length-1 (valid indices are 0 to max_length-1)
        positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, 0, self.max_length - 1)
        

        # print(input_ids)
        token_embeds = self.token_embedding(input_ids)
        
        
        
        pos_embeds = self.position_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        # print('summation ok')
        embeddings = self.dropout(embeddings)
        # print('dropout ok')
        # print("embeddings.shape", embeddings.shape)
        # Create key padding mask: True for tokens to ignore
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        transformer_out = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        # print("transformer_out.shape",transformer_out.shape)
        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            summed = torch.sum(transformer_out * mask, dim=1)
            lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
            pooled = summed / lengths
        else:
            pooled = transformer_out.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

    def forward_with_embeddings(self, embeddings, attention_mask, target_class=None):
        # Truncate if needed
        if embeddings.size(1) > self.max_length:
            embeddings = embeddings[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        batch_size, seq_len, _ = embeddings.size()
        # Clamp seq_len to max_length
        seq_len = min(seq_len, self.max_length)
        # Ensure positions don't exceed max_length-1
        positions = torch.arange(0, seq_len, device=embeddings.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, 0, self.max_length - 1)
        pos_embeds = self.position_embedding(positions)
        combined = embeddings + pos_embeds
        combined = self.dropout(combined)
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        transformer_out = self.transformer_encoder(combined, src_key_padding_mask=src_key_padding_mask)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            summed = torch.sum(transformer_out * mask, dim=1)
            lengths = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1)
            pooled = summed / lengths
        else:
            pooled = transformer_out.mean(dim=1)
        logits = self.classifier(pooled)
        if target_class is not None:
            return logits[:, target_class]
        return logits
    def forward_with_embeddings_soft(self, embeddings, target_class=None):
        """
        Used for inference with hidden representations (`h`) from GPT-2.
        - Assumes `embeddings` is already a representation of the text.
        - Does NOT use token or positional embeddings.
        """
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(1)  # Reshape to [batch_size, 1, embed_dim] for TransformerEncoder

        combined = self.dropout(embeddings)
        transformer_out = self.transformer_encoder(combined).squeeze(1)  # Remove sequence dimension

        logits = self.classifier(transformer_out)
        if target_class is not None:
            return logits[:, target_class]
        return logits