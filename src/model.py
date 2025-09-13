import numpy as np
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    A simple attention model.
    """
    def __init__(self, *, tokens_in, tokens_out, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.pos_embed_k = config.pos_embed_k             # use positional embedding of dimension 2*k
        full_embed_dim = embed_dim + 2 * self.pos_embed_k # the dimension of the activations
        num_heads = config.num_heads                      # num of attention heads
        dropout = 0.0                                     # after initial experiments, not using dropout
        attn_layers = config.attn_layers
        self.embed = nn.Embedding(tokens_in, embed_dim)
        
        self.enc_attn = nn.ModuleList()
        self.enc_linear = nn.ModuleList()
        self.dec_self_attn = nn.ModuleList()
        self.dec_other_attn = nn.ModuleList()
        self.dec_linear = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for _ in range(attn_layers):
            self.enc_attn.append(nn.MultiheadAttention(full_embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.enc_linear.append(nn.Linear(full_embed_dim, full_embed_dim))
            self.dec_self_attn.append(nn.MultiheadAttention(full_embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.dec_other_attn.append(nn.MultiheadAttention(full_embed_dim, num_heads, dropout=dropout, batch_first=True))
            self.dec_linear.append(nn.Linear(full_embed_dim, full_embed_dim))
            self.layer_norm.append(nn.LayerNorm((full_embed_dim,)))
        self.relu = nn.ReLU()
        self.decode = nn.Linear(full_embed_dim, tokens_out)

    @staticmethod
    def get_attn_mask(n):
        return torch.tensor(np.arange(n)[:, np.newaxis] < np.arange(n)[np.newaxis, :])

    def get_positional_embeddings(self, l):
        # output has shape (l, 2*k)
        k = self.pos_embed_k
        output = torch.zeros(2*k, l)
        scale = np.power(l, 1.0 / k)
        for i in range(k):
            scaled_range = scale ** i * torch.arange(l, dtype=torch.float64) * torch.pi / l
            output[2 * i] = torch.cos(scaled_range)
            output[2 * i + 1] = torch.sin(scaled_range)
        return output.transpose(0, 1)

    def forward(self, x, y):
        # x1: (B, L, embed_dim)
        x1 = self.embed(x)
        # x_pos: (L, 2*k)
        x_pos = self.get_positional_embeddings(x1.shape[-2])
        if len(x1.shape) == 3:
            x_pos = torch.unsqueeze(x_pos, 0).expand(x1.shape[0], -1, -1)
        x1 = torch.cat((x_pos, x1), dim = -1)
        
        for i, attention in enumerate(self.enc_attn):
            # x2: (B, L, embed_dim)
            x2, _ = attention(x1, x1, x1)
            x2 = self.relu(x2)
            x2 = self.enc_linear[i](x2)
            x2 = self.relu(x2)
            x2 = x2 + x1
            x1 = x2
        x_encoded = x2
        
        y1 = self.embed(y)
        # y_pos: (L_y, 2*k)
        y_pos = self.get_positional_embeddings(y1.shape[-2])
        if len(y1.shape) == 3:
            y_pos = torch.unsqueeze(y_pos, 0).expand(y1.shape[0], -1, -1)
        y1 = torch.cat((y_pos, y1), dim = -1)

        for i, (self_attention, other_attention) in enumerate(zip(self.dec_self_attn, self.dec_other_attn)):
            y2, _ = self_attention(y1, y1, y1, attn_mask=self.get_attn_mask(y1.shape[-2]))
            y2 = self.relu(y2)
            y2 = y2 + y1
            y3, _ = other_attention(y2, x_encoded, x_encoded)
            y3 = self.relu(y3)
            y3 = self.dec_linear[i](y3)
            y3 = self.relu(y3)
            y3 = y3 + y2
            y1 = self.layer_norm[i](y3)
        logits = self.decode(y1)
        return logits

    def n_params(self):
        return sum([p.numel() for p in self.parameters()])
    
