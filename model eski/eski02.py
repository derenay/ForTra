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

class FormationTransformer(nn.Module):
    def __init__(self, 
                 coord_dim=2, 
                 class_vocab_size=10, 
                 class_embed_dim=16, 
                 embed_dim=128, 
                 num_heads=8, 
                 num_layers=8, 
                 num_formations=10, 
                 dropout=0.1):
        """
        coord_dim: Koordinat boyutu (örneğin, x, y → 2)
        class_vocab_size: Sınıf token sayısı (örneğin 'tank', 'ifv' vs.)
        class_embed_dim: Sınıf embedding boyutu
        embed_dim: Transformer'ın gizli boyutu (fusion sonrası hedef boyut)
        num_formations: Formasyon sınıfı sayısı (örneğin, Line, Wedge, Vee, Echelon Right, Herringbone, Coil, Platoon, Staggered Column, Echelon, Column)
        dropout: Dropout oranı
        num_layers: Transformer encoder katman sayısı
        """
        super(FormationTransformer, self).__init__()
        
        # Ayarlamalar:
        # Toplam embed_dim = 128
        # Biz üç branch kullanacağız:
        # 1. Koordinat branch: boyut = dim_coord (örneğin, 128 // 3 = 42)
        # 2. Sınıf branch: boyut = dim_class (ayarıp, 42 olarak belirleyelim)
        # 3. Relative branch: geri kalan boyut = 128 - (42+42) = 44
        self.dim_coord = embed_dim // 3       # 42
        self.dim_class = embed_dim // 3       # 42
        self.dim_rel = embed_dim - (self.dim_coord + self.dim_class)  # 44

        # Koordinat embedding: [batch, seq_len, 2] -> [batch, seq_len, dim_coord]
        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        
        # Sınıf embedding: önce embedding, sonra projeksiyon: [batch, seq_len] -> [batch, seq_len, dim_class]
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        
        # Relative embedding: hesaplanan relative koordinatlar [batch, seq_len, 2] -> [batch, seq_len, dim_rel]
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel)
        
        # Fusion: concatenate üç branch çıktılarını direkt elde ettiğimiz toplam boyut embed_dim (128)
        # Artık fusion sonrası boyut zaten embed_dim olduğundan ek bir projeksiyona gerek duymayabiliriz.
        
        # Positional encoding (batch_first=True)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder bloğu (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling sonrası norm
        self.norm = nn.LayerNorm(embed_dim)
        # Formasyon sınıflandırması için çıkış katmanı
        self.fc = nn.Linear(embed_dim, num_formations)
        
    def forward(self, coords, class_tokens):
        """
        coords: [batch, seq_len, 2] - normalize edilmiş koordinatlar
        class_tokens: [batch, seq_len] - nesne sınıfı index'leri
        """
        # Koordinat branch
        coord_emb = self.coord_fc(coords)  # [batch, seq_len, dim_coord]
        
        # Sınıf branch
        class_emb = self.class_embedding(class_tokens)  # [batch, seq_len, class_embed_dim]
        class_emb = self.class_fc(class_emb)              # [batch, seq_len, dim_class]
        
        # Relative branch: Her örnekteki koordinatların, o örneğin ortalamasıyla farkı
        # Burada, batch bazında mean hesaplanır (her örneğin merkezi)
        center = coords.mean(dim=1, keepdim=True)  # [batch, 1, 2]
        rel_coords = coords - center               # [batch, seq_len, 2]
        rel_emb = self.rel_fc(rel_coords)          # [batch, seq_len, dim_rel]
        
        # Fusion: branch çıktıları concatenate edilir
        fused_emb = torch.cat([coord_emb, class_emb, rel_emb], dim=-1)  # [batch, seq_len, embed_dim]
        
        # Positional encoding
        x = self.pos_encoder(fused_emb)  # [batch, seq_len, embed_dim]
        
        # Transformer encoder (batch_first=True)
        x = self.transformer_encoder(x)  # [batch, seq_len, embed_dim]
        
        # Global pooling: ortalama alınır
        x = x.mean(dim=1)  # [batch, embed_dim]
        x = self.norm(x)
        out = self.fc(x)   # [batch, num_formations]
        return out
