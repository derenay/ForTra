import torch
import torch.nn as nn
import math

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Adapter(nn.Module):
    def __init__(self, d_model, adapter_dim=64):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, d_model)
        )
    def forward(self, x):
        return self.adapter(x)

class GraphormerMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphormerMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
    def forward(self, x, edge_bias=None, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if edge_bias is not None:
            attn_scores = attn_scores + edge_bias.unsqueeze(1)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_probs = self.attn_dropout(attn_scores.softmax(dim=-1))
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class GraphormerTransformerLayer(nn.TransformerEncoderLayer):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_adapter=False, adapter_dim=64):
        super(GraphormerTransformerLayer, self).__init__(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = Adapter(embed_dim, adapter_dim)

    def forward(self, x, edge_bias=None, attn_mask=None, **kwargs):
        # Ekstra gelen keyword argümanlar kwargs içinde yakalanır (ör. is_causal) ve göz ardı edilir.
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        if self.use_adapter:
            ffn_out = ffn_out + self.adapter(ffn_out)
        x = x + ffn_out
        return x

class HierarchicalFormationTransformer(nn.Module):
    def __init__(self, 
                 coord_dim=2, 
                 class_vocab_size=10, 
                 class_embed_dim=16, 
                 stage_dims=[512, 384, 256], 
                 num_heads=8, 
                 num_layers=[4, 4, 4], 
                 num_formations=10, 
                 dropout_stages=[0.5, 0.3, 0.2], 
                 use_adapter=True, 
                 adapter_dim=64, 
                 pos_type='learnable'):
        super(HierarchicalFormationTransformer, self).__init__()

        initial_dim = stage_dims[0]
        self.dim_coord = initial_dim // 3
        self.dim_class = initial_dim // 3
        self.dim_rel = initial_dim - (self.dim_coord + self.dim_class)

        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel)

        if pos_type == 'learnable':
            self.pos_encoder = LearnablePositionalEmbedding(max_len=5000, d_model=initial_dim)
        else:
            self.pos_encoder = PositionalEncoding(initial_dim, dropout_stages[0])

        # Stage 1: Transformer Encoder (Artık self_attn içeriyor)
        encoder_layer1 = GraphormerTransformerLayer(initial_dim, num_heads, dropout_stages[0], use_adapter, adapter_dim)
        self.encoder_stage1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers[0])

        # Stage 2: 512 -> 384
        self.proj_stage2 = nn.Sequential(
            nn.Linear(initial_dim, stage_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_stages[1])
        )
        encoder_layer2 = GraphormerTransformerLayer(stage_dims[1], num_heads, dropout_stages[1], use_adapter, adapter_dim)
        self.encoder_stage2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers[1])

        # Stage 3: 384 -> 256
        self.proj_stage3 = nn.Sequential(
            nn.Linear(stage_dims[1], stage_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_stages[2])
        )
        encoder_layer3 = GraphormerTransformerLayer(stage_dims[2], num_heads, dropout_stages[2], use_adapter, adapter_dim)
        self.encoder_stage3 = nn.TransformerEncoder(encoder_layer3, num_layers=num_layers[2])

        self.norm = nn.LayerNorm(stage_dims[2])
        self.fc = nn.Linear(stage_dims[2], num_formations)

    def compute_edge_bias(self, coords):
        dist = torch.cdist(coords, coords, p=2)
        norm_factor = dist.mean(dim=(-1,-2), keepdim=True) + 1e-6
        edge_bias = -dist / norm_factor
        return edge_bias  

    def forward(self, coords, class_tokens):
        coord_emb = self.coord_fc(coords)
        class_emb = self.class_embedding(class_tokens)
        class_emb = self.class_fc(class_emb)
        center = coords.mean(dim=1, keepdim=True)
        rel_coords = coords - center
        rel_emb = self.rel_fc(rel_coords)

        fused_emb = torch.cat([coord_emb, class_emb, rel_emb], dim=-1)
        x = self.pos_encoder(fused_emb)
        edge_bias = self.compute_edge_bias(coords)

        x = self.encoder_stage1(x)
        x = self.proj_stage2(x)
        x = self.encoder_stage2(x)
        x = self.proj_stage3(x)
        x = self.encoder_stage3(x)

        x = x.mean(dim=1)
        x = self.norm(x)
        out = self.fc(x)
        return out


