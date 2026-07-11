"""
Patient representation module: a learnable [PAT] token attends over a donor's
cell embeddings via multi-head cross-attention to produce a fixed-size,
donor-count-invariant patient representation (Methods 9.8). Also used as the
attention-based cell explainer (Methods 9.9): per-layer attention weights from
[PAT] to each cell give a per-cell importance score.
"""
import torch
import torch.nn as nn

ATTENTION_HEADS = 4
ATTENTION_LAYERS = 2
ATTENTION_DROPOUT = 0.1


class LinearProbe(nn.Module):
    """Single-layer linear probe, used as the simplest cell-level baseline head."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for PatientAggregator that returns attention weights."""

    def __init__(self, embed_dim, num_heads=ATTENTION_HEADS, dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_token, cell_embeddings, key_padding_mask=None):
        # cls_token: (B, 1, D), cell_embeddings: (B, N, D)
        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask
        attn_output, attn_weights = self.attn(
            cls_token, cell_embeddings, cell_embeddings,
            key_padding_mask=key_padding_mask, need_weights=True,
        )
        cls_token = cls_token + self.dropout(attn_output)
        cls_token = self.norm(cls_token)
        return cls_token, attn_weights


class PatientAggregator(nn.Module):
    """Aggregates a donor's cell embeddings into one patient representation via
    a learnable [PAT] token and stacked cross-attention layers (Methods 9.8)."""

    def __init__(self, embed_dim, num_heads=ATTENTION_HEADS, num_layers=ATTENTION_LAYERS, dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, cell_embeddings, attention_mask=None, return_attention=False):
        B = cell_embeddings.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        attention_weights = []
        for layer in self.attn_layers:
            cls_tokens, attn_weights = layer(cls_tokens, cell_embeddings, key_padding_mask=attention_mask)
            if return_attention:
                attention_weights.append(attn_weights)
        result = cls_tokens.squeeze(1)
        if return_attention:
            return result, attention_weights
        return result


class AttentionClassifier(nn.Module):
    """PatientAggregator-based donor-level classifier."""

    def __init__(self, input_dim, num_classes, num_heads=ATTENTION_HEADS, num_layers=ATTENTION_LAYERS, dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.aggregator = PatientAggregator(input_dim, num_heads, num_layers, dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            patient_embeddings, attention_weights = self.aggregator(x, attention_mask=mask, return_attention=True)
            return self.classifier(patient_embeddings), attention_weights
        patient_embeddings = self.aggregator(x, attention_mask=mask)
        return self.classifier(patient_embeddings)


class AttentionRegressor(nn.Module):
    """PatientAggregator-based donor-level regressor."""

    def __init__(self, input_dim, num_heads=ATTENTION_HEADS, num_layers=ATTENTION_LAYERS, dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.aggregator = PatientAggregator(input_dim, num_heads, num_layers, dropout)
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        patient_embeddings = self.aggregator(x, attention_mask=mask)
        return self.regressor(patient_embeddings).squeeze(-1)
