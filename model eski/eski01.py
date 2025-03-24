
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
        pe[:, 0::2] = torch.sin(position * div_term)  # sin için
        pe[:, 1::2] = torch.cos(position * div_term)  # cos için
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FormationTransformer(nn.Module):
    def __init__(self, coord_dim=2, class_vocab_size=10, class_embed_dim=4, embed_dim=128, 
                 num_heads=4, num_layers=4, num_classes=10, dropout=0.1):
        """
        coord_dim: Koordinat boyutu (örneğin, x, y → 2)
        class_vocab_size: Sınıf token sayısı (örneğin 'tank', 'ifv' vs.) – genişletilebilir
        class_embed_dim: Sınıf embedding boyutu
        embed_dim: Transformer'ın gizli boyutu
        num_classes: Formasyon sınıfı sayısı (örneğin, V, line, A, wall, other)
        dropout: Dropout oranı
        num_layers: Transformer encoder katman sayısı (şimdilik 8 katman kullanıyoruz)
        """
        super(FormationTransformer, self).__init__()
        
        # Konum bilgilerini embed etmek için lineer katman
        self.coord_fc = nn.Linear(coord_dim, embed_dim)
        # Sınıf bilgilerini embed etmek için nn.Embedding
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        # Sınıf embedding'ini embed_dim boyutuna çıkarmak için lineer katman
        self.class_fc = nn.Linear(class_embed_dim, embed_dim)
        
        # Positional encoding ekleyerek Transformer'ın konumsal bilgileri daha iyi öğrenmesini sağlıyoruz
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder bloğu, katman sayısını 8 olarak ayarladık
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sequence pooling sonrası çıkan vektörün normalizasyonu
        self.norm = nn.LayerNorm(embed_dim)
        # Formasyon sınıflandırması için tam bağlı katman
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, coords, class_tokens):
        """
        coords: [batch, seq_len, 2] - normalize edilmiş koordinatlar
        class_tokens: [batch, seq_len] - nesne sınıfı index'leri
        """
        # Koordinat ve sınıf bilgilerini ayrı ayrı embed ediyoruz
        coord_emb = self.coord_fc(coords)                   # [batch, seq_len, embed_dim]
        class_emb = self.class_embedding(class_tokens)        # [batch, seq_len, class_embed_dim]
        class_emb = self.class_fc(class_emb)                  # [batch, seq_len, embed_dim]
        
        # İki embedding'i topluyoruz: Bu, her nesne için birleşik bir temsil sağlar
        x = coord_emb + class_emb                             # [batch, seq_len, embed_dim]
        
        # Positional encoding ekle
        x = self.pos_encoder(x)                               # [batch, seq_len, embed_dim]
        
        # Transformer'ın beklentisi: [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)                                 # [batch, seq_len, embed_dim]
        
        # Sequence pooling: Ortalama alarak global formasyon vektörü elde ediyoruz
        x = x.mean(dim=1)                                    # [batch, embed_dim]
        x = self.norm(x)
        out = self.fc(x)                                     # [batch, num_classes]
        return out
