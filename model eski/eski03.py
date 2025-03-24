import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, Transformer'a ek konumsal bilgi sağlar.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin değerleri
        pe[:, 1::2] = torch.cos(position * div_term)  # cos değerleri
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class GraphormerMultiheadAttention(nn.Module):
    """
    Graphormer tarzı multihead self-attention.
    Edge encoding ekleyerek nesneler arası ilişkileri modelleyebilir.
    """
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
    
    def forward(self, x, edge_weights=None, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        if edge_weights is not None:
            attn_scores = attn_scores + edge_weights  # edge_weights boyutunun uygun olduğundan emin olunmalıdır.
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_probs = self.attn_dropout(attn_scores.softmax(dim=-1))
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class GraphormerTransformerLayer(nn.Module):
    """
    Graphormer tarzı Transformer encoder katmanı.
    Attention sonrası residual bağlantılar, layer norm ve FFN içerir.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphormerTransformerLayer, self).__init__()
        self.attn = GraphormerMultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, edge_weights=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), edge_weights, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalFormationTransformer(nn.Module):
    def __init__(self, 
                 coord_dim=2, 
                 class_vocab_size=10, 
                 class_embed_dim=16, 
                 stage_dims=[512, 384, 256], 
                 num_heads=8, 
                 num_layers=[8, 8, 8], 
                 num_formations=10, 
                 dropout_stages=[0.5, 0.3, 0.2]):
        """
        coord_dim: Koordinat boyutu (örn. x, y → 2)
        class_vocab_size: Sınıf token sayısı
        class_embed_dim: Sınıf embedding boyutu
        stage_dims: Her hiyerarşik aşamadaki embedding boyutları, örn. [512, 384, 256]
        num_heads: Her aşama için Transformer encoder head sayısı (aynı kabul edilir)
        num_layers: Her aşamada kullanılacak encoder katman sayıları, örn. [4, 4, 4]
        num_formations: Formasyon sınıfı sayısı
        dropout_stages: Her aşama için dropout oranları (dinamik), örn. [0.5, 0.3, 0.2]
        """
        super(HierarchicalFormationTransformer, self).__init__()
        
        initial_dim = stage_dims[0]
        self.dim_coord = initial_dim // 3
        self.dim_class = initial_dim // 3
        self.dim_rel = initial_dim - (self.dim_coord + self.dim_class)
        
        # Branch'ler: koordinat, sınıf ve relative bilgileri ayrı ayrı embed ediliyor.
        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel)
        
        # Positional encoding: ilk aşama boyutunda
        self.pos_encoder = PositionalEncoding(initial_dim, dropout_stages[0])
        
        # Stage 1: Transformer encoder katmanları (512 boyutlu)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=initial_dim, nhead=num_heads, dropout=dropout_stages[0], batch_first=True)
        self.encoder_stage1 = nn.TransformerEncoder(encoder_layer1, num_layers=num_layers[0])
        
        # Stage 2: Boyut düşürme (512 -> 384) + Encoder katmanları
        self.proj_stage2 = nn.Sequential(
            nn.Linear(initial_dim, stage_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_stages[1])
        )
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=stage_dims[1], nhead=num_heads, dropout=dropout_stages[1], batch_first=True)
        self.encoder_stage2 = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers[1])
        
        # Stage 3: Boyut düşürme (384 -> 256) + Encoder katmanları
        self.proj_stage3 = nn.Sequential(
            nn.Linear(stage_dims[1], stage_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_stages[2])
        )
        encoder_layer3 = nn.TransformerEncoderLayer(d_model=stage_dims[2], nhead=num_heads, dropout=dropout_stages[2], batch_first=True)
        self.encoder_stage3 = nn.TransformerEncoder(encoder_layer3, num_layers=num_layers[2])
        
        # Global pooling sonrası norm ve final sınıflandırma katmanı
        self.norm = nn.LayerNorm(stage_dims[2])
        self.fc = nn.Linear(stage_dims[2], num_formations)
    
    def compute_edge_weights(self, coords):
        """
        Edge encoding: Her nesne çifti arasındaki Öklid mesafelerini hesaplar ve negatif normalize eder.
        coords: [batch, seq_len, 2]
        """
        dist = torch.cdist(coords, coords, p=2)  # [batch, seq_len, seq_len]
        norm_factor = dist.mean(dim=(-1,-2), keepdim=True) + 1e-6
        edge_weights = -dist / norm_factor
        return edge_weights  # [batch, seq_len, seq_len]
    
    def forward(self, coords, class_tokens):
        """
        coords: [batch, seq_len, 2] - normalize edilmiş koordinatlar
        class_tokens: [batch, seq_len] - nesne sınıfı index'leri
        """
        # Hesapla: Branch'lerin çıktıları
        coord_emb = self.coord_fc(coords)  # [batch, seq_len, dim_coord]
        class_emb = self.class_embedding(class_tokens)  # [batch, seq_len, class_embed_dim]
        class_emb = self.class_fc(class_emb)              # [batch, seq_len, dim_class]
        center = coords.mean(dim=1, keepdim=True)           # [batch, 1, 2]
        rel_coords = coords - center                        # [batch, seq_len, 2]
        rel_emb = self.rel_fc(rel_coords)                   # [batch, seq_len, dim_rel]
        
        # Fusion: Concatenate branch çıktıları → [batch, seq_len, initial_dim]
        fused_emb = torch.cat([coord_emb, class_emb, rel_emb], dim=-1)
        
        # Positional encoding
        x = self.pos_encoder(fused_emb)
        
        # Stage 1
        x = self.encoder_stage1(x)
        
        # Stage 2
        x = self.proj_stage2(x)
        x = self.encoder_stage2(x)
        
        # Stage 3
        x = self.proj_stage3(x)
        x = self.encoder_stage3(x)
        
        # Global pooling: tüm sequence'in ortalamasını al
        x = x.mean(dim=1)  # [batch, final_dim]
        # Layer normalization
        x = self.norm(x)
        # Final linear katman (sınıflandırma)
        out = self.fc(x)   # [batch, num_formations]
        return out

