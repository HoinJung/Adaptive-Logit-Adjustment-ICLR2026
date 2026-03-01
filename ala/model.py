import torch.nn as nn
import torch.nn.functional as F
import torch


class SimpleTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=2, num_heads=4, max_length=128, num_classes=2, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        batch_size, seq_len = input_ids.size()
        seq_len = min(seq_len, self.max_length)
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length

        vocab_size = self.token_embedding.num_embeddings
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, 0, self.max_length - 1)

        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)

        if attention_mask is not None:
            if attention_mask.size(1) != input_ids.size(1):
                attention_mask = attention_mask[:, :input_ids.size(1)]
            attention_mask = attention_mask.long()
            attention_mask = torch.clamp(attention_mask, 0, 1)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        transformer_out = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

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
        if embeddings.size(1) > self.max_length:
            embeddings = embeddings[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        batch_size, seq_len, _ = embeddings.size()
        seq_len = min(seq_len, self.max_length)
        if seq_len > self.max_length:
            embeddings = embeddings[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length

        positions = torch.arange(0, seq_len, device=embeddings.device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, 0, self.max_length - 1)
        pos_embeds = self.position_embedding(positions)
        combined = embeddings + pos_embeds
        combined = self.dropout(combined)

        if attention_mask is not None:
            if attention_mask.size(1) != embeddings.size(1):
                attention_mask = attention_mask[:, :embeddings.size(1)]
            attention_mask = attention_mask.long()
            attention_mask = torch.clamp(attention_mask, 0, 1)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
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
