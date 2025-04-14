

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, List, Dict, Any, Tuple

# --- Positional Encoding Modules ---

class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embedding layer.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Embedding dimension.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Use nn.Parameter so gradients are computed for these embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor with added positional embeddings,
                          shape [batch_size, seq_len, d_model].
        """
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # Slice embeddings up to the current sequence length and add
        # Broadcasting adds the same positional embedding across the batch.
        return x + self.pos_embedding[:, :seq_len, :]

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding layer.

    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position indices: [max_len, 1]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term for sine/cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Create matrix for positional encodings: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (handle cases where d_model is odd)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) # Match shape if d_model is odd

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer, not a model parameter
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor with added positional encodings and dropout,
                          shape [batch_size, seq_len, d_model].
        """
        # Slice 'pe' to match input sequence length and add
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- Adapter Module ---

class Adapter(nn.Module):
    """
    Simple bottleneck Adapter module.

    Args:
        d_model (int): Input and output dimension.
        adapter_dim (int): Hidden dimension of the adapter bottleneck. Default: 64.
        activation (nn.Module): Activation function. Default: nn.GELU.
    """
    def __init__(self, d_model: int, adapter_dim: int = 64, activation: nn.Module = nn.GELU):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            activation(),
            nn.Linear(adapter_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Passes input through the adapter. """
        return self.adapter(x)

# --- Graphormer-style Attention (MODIFIED for Directional Bias) ---

class GraphormerMultiheadAttention(nn.Module):
    # ... (__init__ aynı kalır) ...
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                spatial_bias: Optional[torch.Tensor] = None,     # Shape [B, H, T, S] bekleniyor
                directional_bias: Optional[torch.Tensor] = None, # Shape [B, H, T, S] bekleniyor
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        # ... (projeksiyonlar ve attn_scores hesaplaması aynı) ...
        B, T, E = query.shape
        _, S, _ = key.shape
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # Maskeleme aynı
        if attn_mask is not None:
            if attn_mask.dim() == 2: attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # --- Bias Ekleme (GÜNCELLENDİ) ---
        if T == S: # Self-attention durumu
            if spatial_bias is not None:
                # spatial_bias zaten [B, H, S, S] şeklinde geldiği için unsqueeze YOK
                attn_scores = attn_scores + spatial_bias
            if directional_bias is not None:
                # directional_bias zaten [B, H, S, S] şeklinde geldiği için unsqueeze YOK
                attn_scores = attn_scores + directional_bias

        # ... (softmax, dropout, output hesaplaması aynı) ...
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        output = self.out_proj(attn_output)
        return output, None



# ÖNEMLİ: GraphormerTransformerLayer ve CustomTransformerEncoder sınıflarında
# bir değişiklik yapmaya genellikle gerek yoktur, çünkü onlar sadece bias tensörlerini
# olduğu gibi alt katmana (GraphormerMultiheadAttention'a) iletirler.
# Sadece argüman isimleri ve docstring'ler güncellenebilir.


# --- Custom Transformer Layer and Encoder (MODIFIED for Directional Bias) ---

class GraphormerTransformerLayer(nn.Module):
    """
    Transformer Encoder Layer with Graphormer Attention (Spatial+Directional) and optional Adapter.
    Uses pre-layer normalization (Pre-LN).
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 activation: nn.Module = nn.GELU,
                 use_adapter: bool = False,
                 adapter_dim: int = 64):
        super().__init__()

        # --- Self-Attention Block ---
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = GraphormerMultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout) # Dropout after attention projection

        # --- Feed-Forward Block ---
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation()
        self.dropout_ffn1 = nn.Dropout(dropout) # Dropout after activation
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout) # Dropout after final FFN linear

        # --- Adapter (optional) ---
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapter = Adapter(d_model, adapter_dim, activation=activation)
        else:
            self.adapter = None

    def forward(self,
                src: torch.Tensor,
                spatial_bias: Optional[torch.Tensor] = None,      # Renamed
                directional_bias: Optional[torch.Tensor] = None,  # NEW
                src_mask: Optional[torch.Tensor] = None, # Optional standard attention mask
                src_key_padding_mask: Optional[torch.Tensor] = None # Padding mask
                ) -> torch.Tensor:
        """
        Forward pass for the Transformer layer (Pre-LN structure).

        Args:
            src (torch.Tensor): Input tensor [B, S, E].
            spatial_bias (Optional[torch.Tensor]): Graphormer spatial bias [B, S, S].
            directional_bias (Optional[torch.Tensor]): Graphormer directional bias [B, S, S].
            src_mask (Optional[torch.Tensor]): Standard attention mask [S, S] or [B, S, S].
            src_key_padding_mask (Optional[torch.Tensor]): Padding mask [B, S].

        Returns:
            torch.Tensor: Output tensor [B, S, E].
        """
        # --- Self-Attention Part (Pre-LN) ---
        residual = src
        x_norm = self.norm1(src)
        # Pass BOTH biases to the modified attention mechanism
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, # Self-attention
                                        spatial_bias=spatial_bias,
                                        directional_bias=directional_bias,
                                        key_padding_mask=src_key_padding_mask,
                                        attn_mask=src_mask)
        x = residual + self.dropout1(attn_output) # Add residual after dropout

        # --- Feed-Forward Part (Pre-LN) ---
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.linear2(self.dropout_ffn1(self.activation(self.linear1(x_norm))))

        # Add adapter output to FFN output *before* the main residual connection
        if self.adapter is not None:
            adapter_output = self.adapter(ffn_output) # Apply adapter to FFN output
            ffn_output = ffn_output + adapter_output

        x = residual + self.dropout2(ffn_output) # Add residual after dropout

        return x

class CustomTransformerEncoder(nn.Module):
    """
    A stack of Transformer Encoder Layers that correctly passes spatial and directional biases.
    """
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        # Create deep copies of the encoder layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: torch.Tensor,
                spatial_bias: Optional[torch.Tensor] = None,      # Renamed
                directional_bias: Optional[torch.Tensor] = None,  # NEW
                mask: Optional[torch.Tensor] = None, # Standard attn mask (e.g., causal)
                src_key_padding_mask: Optional[torch.Tensor] = None # Padding mask
                ) -> torch.Tensor:
        """
        Passes input through the stack of layers.

        Args:
            src (torch.Tensor): Input tensor [B, S, E].
            spatial_bias (Optional[torch.Tensor]): Graphormer spatial bias [B, S, S].
            directional_bias (Optional[torch.Tensor]): Graphormer directional bias [B, S, S].
            mask (Optional[torch.Tensor]): Standard attention mask passed to all layers.
            src_key_padding_mask (Optional[torch.Tensor]): Padding mask passed to all layers.

        Returns:
            torch.Tensor: Output tensor [B, S, E].
        """
        output = src
        for layer in self.layers:
            output = layer(output,
                           spatial_bias=spatial_bias,
                           directional_bias=directional_bias,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# --- Main Hierarchical Model (MODIFIED for Directional Bias) ---
class HierarchicalFormationTransformer(nn.Module):
    """
    Öğrenilebilir geometrik kenar bilgileri kullanan ve pozisyonel kodlama
    içermeyen Hiyerarşik Transformer.
    (Args güncellendi: pos_type kaldırıldı, bias parametreleri eklendi)
    """
    def __init__(self,
                 num_formations: int,
                 class_vocab_size: int,
                 class_embed_dim: int = 64,
                 coord_dim: int = 2,
                 direction_dim: int = 1,
                 stage_dims: List[int] = [256, 128], # Örnek: 2 aşama
                 num_heads: int = 8,
                 num_layers: List[int] = [6, 6],     # Örnek: 2 aşama
                 dropout_stages: List[float] = [0.2, 0.1], # Örnek: 2 aşama
                 use_adapter: bool = True,
                 adapter_dim: int = 32,
                 max_len: int = 50, # Max birim sayısı (Embedding size için değil, bias için limit)
                 ffn_ratio: int = 4,
                 # --- Yeni Bias Parametreleri ---
                 num_spatial_bins: int = 16, # Mekansal mesafe kovası sayısı
                 max_distance: float = 1.5,  # Dikkate alınacak maksimum mesafe (normalize koordinatlar için)
                 num_directional_bins: int = 16 # Yönelimsel ilişki kovası sayısı
                 ):
        super().__init__()

        # Parametre kontrolleri
        assert len(stage_dims) == len(num_layers) == len(dropout_stages), \
            "stage_dims, num_layers, and dropout_stages must have the same length."
        assert direction_dim > 0, "direction_dim must be positive"

        self.direction_dim = direction_dim
        self.num_heads = num_heads
        num_stages = len(stage_dims)
        initial_dim = stage_dims[0]

        # --- Girdi Embedding Boyutları ---
        total_parts = 4 # coord, dir, class, rel_coord
        base_dim = initial_dim // total_parts
        remainder = initial_dim % total_parts
        self.dim_coord = base_dim + remainder
        self.dim_dir = base_dim
        self.dim_class = base_dim
        self.dim_rel = base_dim
        current_total = self.dim_coord + self.dim_dir + self.dim_class + self.dim_rel
        if current_total != initial_dim:
             self.dim_coord += (initial_dim - current_total)
        assert self.dim_coord + self.dim_dir + self.dim_class + self.dim_rel == initial_dim, \
               f"Input dimension allocation error."

        # --- Girdi Projeksiyon Katmanları ---
        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        self.direction_fc = nn.Linear(self.direction_dim, self.dim_dir)
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel)

        # --- Pozisyonel Kodlama (KALDIRILDI) ---
        # self.pos_encoder = ... Artık yok

        # --- Öğrenilebilir Kenar Bilgisi (Bias) Embedding'leri ---
        self.num_spatial_bins = num_spatial_bins
        self.max_distance = max_distance
        # padding_idx=0: Kova indeksi 0 özel durumlar (örn. kendi kendine mesafe) için kullanılır.
        # Boyut num_heads: Her başlık kendi bias'ını öğrenebilir.
        self.spatial_bias_embedding = nn.Embedding(num_spatial_bins + 1, num_heads, padding_idx=0)

        self.num_directional_bins = num_directional_bins
        self.directional_bias_embedding = nn.Embedding(num_directional_bins + 1, num_heads, padding_idx=0)
        # Embedding'leri sıfırla başlatmak iyi olabilir (özellikle padding_idx)
        nn.init.zeros_(self.spatial_bias_embedding.weight)
        nn.init.zeros_(self.directional_bias_embedding.weight)


        # --- Transformer Aşamaları ---
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        current_dim = initial_dim
        for i in range(num_stages):
            # ... (projeksiyon ve stage oluşturma mantığı aynı kalır) ...
             stage_dim = stage_dims[i]
             stage_dropout = dropout_stages[i]
             stage_num_layers = num_layers[i]
             ff_dim = ffn_ratio * stage_dim

             if i > 0:
                 proj = nn.Sequential(
                     nn.Linear(current_dim, stage_dim), nn.GELU(), nn.Dropout(stage_dropout)
                 )
                 self.projections.append(proj)
             else:
                 self.projections.append(nn.Identity())

             # Dikkat: GraphormerTransformerLayer'ın içindeki GraphormerMultiheadAttention'ın
             # bias'ları nasıl aldığını (unsqueeze olmadan) güncellediğimizden emin olmalıyız.
             encoder_layer = GraphormerTransformerLayer(
                 d_model=stage_dim, nhead=num_heads, dim_feedforward=ff_dim,
                 dropout=stage_dropout, activation=nn.GELU,
                 use_adapter=use_adapter, adapter_dim=adapter_dim
             )
             encoder_stage = CustomTransformerEncoder(
                 encoder_layer, num_layers=stage_num_layers
             )
             self.stages.append(encoder_stage)
             current_dim = stage_dim

        # --- Çıkış Katmanları ---
        last_stage_dim = stage_dims[-1]
        self.pooling_weights_fc = nn.Linear(last_stage_dim, 1)
        self.output_norm = nn.LayerNorm(last_stage_dim)
        self.output_fc = nn.Linear(last_stage_dim, num_formations)

    def _discretize_distance(self, distances: torch.Tensor) -> torch.Tensor:
        """Mesafeleri ayrık kovalara böler."""
        bins = torch.linspace(0, self.max_distance, self.num_spatial_bins, device=distances.device)
        clipped_distances = torch.clamp(distances, max=self.max_distance)
        # Kovalar 1'den başlasın diye +1 ekliyoruz
        distance_bins = torch.bucketize(clipped_distances, bins, right=True) + 1

        # <<< DÜZELTME: İndeksleri üst sınıra klipsle >>>
        # Geçerli maksimum indeks num_spatial_bins'tir (çünkü embedding boyutu num_spatial_bins + 1)
        distance_bins = torch.clamp(distance_bins, max=self.num_spatial_bins)

        # Köşegeni padding indeksi (0) yap
        if distances.shape[-1] == distances.shape[-2]:
             self.fill_diagonal_(distance_bins, 0)

        return distance_bins.long()

    def _discretize_direction(self, relative_directions: torch.Tensor) -> torch.Tensor:
        """Göreceli yönleri (örn. cos sim) ayrık kovalara böler."""
        # [-1, 1] aralığını [0, 1]'e normalize et
        normalized_rel_dirs = (relative_directions + 1.0) / 2.0
        bins = torch.linspace(0, 1.0, self.num_directional_bins, device=relative_directions.device)
         # Kovalar 1'den başlasın diye +1 ekliyoruz
        direction_bins = torch.bucketize(normalized_rel_dirs, bins, right=True) + 1

        # <<< DÜZELTME: İndeksleri üst sınıra klipsle >>>
        # Geçerli maksimum indeks num_directional_bins'tir
        direction_bins = torch.clamp(direction_bins, max=self.num_directional_bins)

        # Köşegeni padding indeksi (0) yap
        if relative_directions.shape[-1] == relative_directions.shape[-2]:
            self.fill_diagonal_(direction_bins, 0)

        return direction_bins.long()

    # --- Yeni Öğrenilebilir Bias Hesaplama Metotları ---
    def _compute_learnable_spatial_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """Öğrenilebilir mekansal bias hesaplar."""
        B, S, _ = coords.shape
        if S <= 1:
            # embedding(0) padding_idx olduğu için num_heads boyutunda sıfır döner
            return self.spatial_bias_embedding(torch.zeros((B, S, S), dtype=torch.long, device=coords.device))

        # 1. Mesafeleri Hesapla
        distances = torch.cdist(coords, coords, p=2) # [B, S, S]

        # 2. Mesafeleri Kovalara Ayır
        distance_bins = self._discretize_distance(distances) # [B, S, S], integer indices

        # 3. Embedding Lookup
        # [B, S, S] -> [B, S, S, num_heads]
        spatial_bias = self.spatial_bias_embedding(distance_bins)

        # 4. Boyutları Dikkat Mekanizmasına Uygun Hale Getir: [B, H, S, S]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)

        return spatial_bias

    def _compute_learnable_directional_bias(self, directions: torch.Tensor) -> torch.Tensor:
        """Öğrenilebilir yönelimsel bias hesaplar."""
        B, S, D = directions.shape
        if S <= 1:
            # embedding(0) padding_idx olduğu için num_heads boyutunda sıfır döner
            return self.directional_bias_embedding(torch.zeros((B, S, S), dtype=torch.long, device=directions.device))

        # 1. Göreceli Yönü Hesapla
        if self.direction_dim == 1:
            directions_rad = directions * (2 * math.pi)
            dir_diff = directions_rad.unsqueeze(2) - directions_rad.unsqueeze(1)
            relative_directions = torch.cos(dir_diff).squeeze(-1) # [B, S, S]
        elif self.direction_dim == 2:
            directions_norm = F.normalize(directions, p=2, dim=-1, eps=1e-6)
            relative_directions = torch.bmm(directions_norm, directions_norm.transpose(1, 2)) # [B, S, S]
        else:
            return self.directional_bias_embedding(torch.zeros((B, S, S), dtype=torch.long, device=directions.device))

        # 2. Göreceli Yönleri Kovalara Ayır
        direction_bins = self._discretize_direction(relative_directions) # [B, S, S], integer indices

        # 3. Embedding Lookup
        # [B, S, S] -> [B, S, S, num_heads]
        directional_bias = self.directional_bias_embedding(direction_bins)

        # 4. Boyutları Dikkat Mekanizmasına Uygun Hale Getir: [B, H, S, S]
        directional_bias = directional_bias.permute(0, 3, 1, 2)

        return directional_bias

    # Helper to fill diagonal for bias calculation functions
    # Needed because torch.Tensor doesn't have fill_diagonal_ directly
    def fill_diagonal_(self, tensor: torch.Tensor, value: Any):
        """Fills the diagonals of the last two dimensions of a tensor."""
        if tensor.dim() < 2:
            return
        B = tensor.shape[:-2] # Get batch dimensions
        S = tensor.shape[-1]
        if tensor.shape[-1] != tensor.shape[-2]: # Check if last two dims are square
             return
        diag_mask = torch.eye(S, dtype=torch.bool, device=tensor.device)
        # Expand mask to match batch dimensions
        for _ in range(tensor.dim() - 2):
            diag_mask = diag_mask.unsqueeze(0)
        diag_mask = diag_mask.expand_as(tensor) # Expand correctly
        tensor.masked_fill_(diag_mask, value)


    # --- Güncellenmiş Forward Metodu ---
    def forward(self,
                coords: torch.Tensor,
                class_tokens: torch.Tensor,
                directions: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Öğrenilebilir biaslar kullanan ve pozisyonel kodlama içermeyen forward pass.
        """
        B, S, Cdim = coords.shape
        # Shape kontrolleri
        assert directions.shape == (B, S, self.direction_dim), "Directions shape mismatch"
        assert class_tokens.shape == (B, S), "Class tokens shape mismatch"
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, S), "Padding mask shape mismatch"

        device = coords.device

        # --- 1. Girdi Embedding'leri ---
        coord_emb = F.gelu(self.coord_fc(coords))
        direction_emb = F.gelu(self.direction_fc(directions))
        class_emb = self.class_embedding(class_tokens)
        class_emb = F.gelu(self.class_fc(class_emb))

        # --- Göreceli Koordinatlar ---
        if key_padding_mask is not None:
            keep_mask = ~key_padding_mask.unsqueeze(-1)
            num_non_padded = keep_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            center = (coords * keep_mask).sum(dim=1, keepdim=True) / num_non_padded
            rel_coords = (coords - center) * keep_mask
        else:
            center = coords.mean(dim=1, keepdim=True)
            rel_coords = coords - center
        rel_emb = F.gelu(self.rel_fc(rel_coords))

        # --- 2. Embedding'leri Birleştir ---
        fused_emb = torch.cat([coord_emb, direction_emb, class_emb, rel_emb], dim=-1)

        # --- 3. Pozisyonel Kodlama (KALDIRILDI) ---
        x = fused_emb # Sıra bilgisi yok

        # --- 4. Öğrenilebilir Kenar Bilgilerini Hesapla ---
        # Hesaplanan bias'ların şekli [B, num_heads, S, S] olacak
        spatial_bias = self._compute_learnable_spatial_bias(coords)
        directional_bias = self._compute_learnable_directional_bias(directions)

        # --- 5. Transformer Aşamalarından Geçir ---
        current_input = x
        for i, (proj, stage) in enumerate(zip(self.projections, self.stages)):
            projected_input = proj(current_input)
            # stage -> CustomTransformerEncoder -> GraphormerTransformerLayer
            # Biaslar doğru şekille (artık num_heads boyutu var) iletiliyor
            output = stage(projected_input,
                           spatial_bias=spatial_bias,
                           directional_bias=directional_bias,
                           src_key_padding_mask=key_padding_mask)
            current_input = output

        # --- 6. Havuzlama ve Sınıflandırma ---
        # Dikkat: current_input kullanılmalı
        pool_logits = self.pooling_weights_fc(current_input)
        if key_padding_mask is not None:
            pool_logits = pool_logits.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))
        pool_attn_weights = F.softmax(pool_logits, dim=1)
        pooled_output = (current_input * pool_attn_weights).sum(dim=1)
        pooled_output_norm = self.output_norm(pooled_output)
        out_logits = self.output_fc(pooled_output_norm)

        return out_logits
