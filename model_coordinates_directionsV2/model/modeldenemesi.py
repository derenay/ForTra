

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
    """
    Multi-head self-attention with Graphormer-style spatial and directional edge bias.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability for attention weights. Default: 0.1.
    """
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
                spatial_bias: Optional[torch.Tensor] = None,     # Renamed for clarity
                directional_bias: Optional[torch.Tensor] = None, # Directional bias input
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor [B, T, E].
            key (torch.Tensor): Key tensor [B, S, E].
            value (torch.Tensor): Value tensor [B, S, E].
            spatial_bias (Optional[torch.Tensor]): Graphormer spatial distance bias [B, T, S].
            directional_bias (Optional[torch.Tensor]): Graphormer directional relation bias [B, T, S].
            key_padding_mask (Optional[torch.Tensor]): Mask for key padding [B, S].
            attn_mask (Optional[torch.Tensor]): Additional attention mask [T, S] or [B, T, S].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Attention output [B, T, E], None.
        """
        B, T, E = query.shape
        B_k, S, E_k = key.shape
        B_v, S_v, E_v = value.shape
        assert B == B_k == B_v, "Batch sizes must match"
        assert S == S_v, "Key and Value sequence lengths must match"
        assert E == E_k == E_v, "Embedding dimensions must match"


        # 1. Linear projections + split into heads
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # [B, H, T, S]

        # 3. Apply masks and biases BEFORE softmax
        if attn_mask is not None:
            # Needs correct broadcasting: e.g., [T, S] -> [1, 1, T, S]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: # [B, T, S] -> [B, 1, T, S]
                 attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask # Add mask biases

        if key_padding_mask is not None:
            # key_padding_mask [B, S] -> [B, 1, 1, S]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # --- Add Graphormer Biases ---
        # Ensure biases have shape [B, 1, T, S] for broadcasting over heads
        if T == S: # Apply biases typically only for self-attention (T==S)
            if spatial_bias is not None:
                 attn_scores = attn_scores + spatial_bias.unsqueeze(1)


            if directional_bias is not None:
                 # Hatanın oluştuğu varsayılan satır (197 civarı)
                 attn_scores = attn_scores + directional_bias.unsqueeze(1)

        # 4. Softmax and Dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 5. Weighted sum of values
        attn_output = attn_probs @ v # [B, H, T, D]

        # 6. Concatenate heads and final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None

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
    Hierarchical Transformer enhanced with Directional Edge Bias for formation
    classification using coordinates, class types, and directions.

    Args:
        num_formations (int): Number of output formation classes.
        class_vocab_size (int): Number of unique entity classes.
        class_embed_dim (int): Initial dimension for class embeddings lookup.
        coord_dim (int): Dimension of input coordinates (usually 2). Default: 2.
        direction_dim (int): Dimension of input directions (e.g., 1 for angle, 2 for vector).
                             *** Make sure this matches your input data format! *** Default: 1.
        stage_dims (List[int]): List of embedding dimensions for each transformer stage.
        num_heads (int): Number of attention heads in each stage.
        num_layers (List[int]): List of number of layers for each transformer stage.
        dropout_stages (List[float]): List of dropout rates for each stage.
        use_adapter (bool): Whether to use Adapter modules in transformer layers.
        adapter_dim (int): Hidden dimension for Adapter modules.
        pos_type (str): Type of positional encoding ('learnable' or 'sinusoidal').
        max_len (int): Maximum sequence length for positional encodings.
        ffn_ratio (int): Ratio for feedforward dimension (dim_ffn = ratio * d_model). Default: 4.
    """
    def __init__(self,
                 num_formations: int,
                 class_vocab_size: int,
                 class_embed_dim: int = 64,
                 coord_dim: int = 2,
                 direction_dim: int = 1, # <<< Ensure this matches your data!
                 stage_dims: List[int] = [256, 128, 64], # Example dimensions
                 num_heads: int = 8,
                 num_layers: List[int] = [4, 4, 4],
                 dropout_stages: List[float] = [0.1, 0.1, 0.1],
                 use_adapter: bool = True,
                 adapter_dim: int = 32,
                 pos_type: str = 'learnable',
                 max_len: int = 500, # Max entities in a formation
                 ffn_ratio: int = 4
                 ):
        super().__init__()

        assert len(stage_dims) == len(num_layers) == len(dropout_stages), \
            "stage_dims, num_layers, and dropout_stages must have the same length."
        assert pos_type in ['learnable', 'sinusoidal'], "pos_type must be 'learnable' or 'sinusoidal'"
        assert direction_dim > 0, "direction_dim must be positive"
        self.direction_dim = direction_dim # Store direction dim

        num_stages = len(stage_dims)
        initial_dim = stage_dims[0] # Dimension after initial embedding fusion

        # --- Define dimensions for concatenated input embeddings ---
        total_parts = 4 # coord, dir, class, rel_coord
        base_dim = initial_dim // total_parts
        remainder = initial_dim % total_parts
        self.dim_coord = base_dim + remainder
        self.dim_dir = base_dim
        self.dim_class = base_dim
        self.dim_rel = base_dim
        # Adjust allocation if needed, e.g., give more to class or coords
        current_total = self.dim_coord + self.dim_dir + self.dim_class + self.dim_rel
        if current_total != initial_dim:
             # Handle potential rounding issues if total_parts doesn't divide initial_dim well
             # Add any remaining dimension to the largest part (coord_dim usually)
             self.dim_coord += (initial_dim - current_total)
        assert self.dim_coord + self.dim_dir + self.dim_class + self.dim_rel == initial_dim, \
               f"Input dimension allocation error: {self.dim_coord}+{self.dim_dir}+{self.dim_class}+{self.dim_rel} != {initial_dim}"

        # --- Input Projection Layers ---
        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        self.direction_fc = nn.Linear(self.direction_dim, self.dim_dir) # Use stored direction_dim
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel) # Relative coords are still coord_dim

        # --- Positional Encoding ---
        if pos_type == 'learnable':
            self.pos_encoder = LearnablePositionalEmbedding(max_len=max_len, d_model=initial_dim)
        else: # sinusoidal
            self.pos_encoder = PositionalEncoding(initial_dim, dropout_stages[0], max_len=max_len)

        # --- Transformer Stages ---
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        current_dim = initial_dim

        for i in range(num_stages):
            stage_dim = stage_dims[i]
            stage_dropout = dropout_stages[i]
            stage_num_layers = num_layers[i]
            ff_dim = ffn_ratio * stage_dim

            # Projection layer (except for the first stage)
            if i > 0:
                proj = nn.Sequential(
                    nn.Linear(current_dim, stage_dim),
                    nn.GELU(),
                    nn.Dropout(stage_dropout)
                )
                self.projections.append(proj)
            else:
                self.projections.append(nn.Identity()) # No projection before first stage

            # Transformer Encoder for the current stage (Uses updated layer/encoder)
            encoder_layer = GraphormerTransformerLayer(
                d_model=stage_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=stage_dropout,
                activation=nn.GELU,
                use_adapter=use_adapter,
                adapter_dim=adapter_dim
            )
            encoder_stage = CustomTransformerEncoder(
                encoder_layer,
                num_layers=stage_num_layers
                # No final norm here, applied after pooling
            )
            self.stages.append(encoder_stage)
            current_dim = stage_dim # Update dimension for next projection

        # --- Output Layers ---
        last_stage_dim = stage_dims[-1]
        self.pooling_weights_fc = nn.Linear(last_stage_dim, 1) # For attention pooling
        self.output_norm = nn.LayerNorm(last_stage_dim)
        self.output_fc = nn.Linear(last_stage_dim, num_formations)

    def compute_spatial_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes Graphormer-style spatial bias based on pairwise distances.

        Args:
            coords (torch.Tensor): Coordinates tensor [B, S, coord_dim].

        Returns:
            torch.Tensor: Spatial bias tensor [B, S, S]. Closer points get higher bias.
        """
        B, S, _ = coords.shape
        if S <= 1:
             return torch.zeros((B, S, S), device=coords.device, dtype=coords.dtype)

        dist = torch.cdist(coords, coords, p=2) # Pairwise Euclidean distances

        # Simple negative distance scaling (consider normalization if needed)
        # Smaller distance -> less negative bias -> higher attention score contribution
        spatial_bias = -dist
        # Optional: Normalize or scale differently based on experimentation
        # norm_factor = dist.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        # spatial_bias = -dist / norm_factor
        return spatial_bias

    def compute_directional_bias(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Çiftler arası yön benzerliğine dayalı Graphormer tarzı yönelimsel bias hesaplar.
        direction_dim=1 ise, girdinin 0-1 aralığında normalize edilmiş açı olduğunu
        varsayar ve bunu 0-2pi radyana ölçekleyerek işlem yapar.

        Args:
            directions (torch.Tensor): Yön tensörü [B, S, direction_dim].

        Returns:
            torch.Tensor: Yönelimsel bias tensörü [B, S, S]. Yüksek değerler daha
                          güçlü yönelimsel hizalanma/benzerlik gösterir. S <= 1 ise sıfır döndürür.
        """
        B, S, D = directions.shape
        assert D == self.direction_dim, f"Yön boyutu uyuşmazlığı: Beklenen {self.direction_dim}, Gelen {D}"
        if S <= 1:
            return torch.zeros((B, S, S), device=directions.device, dtype=directions.dtype)

        if self.direction_dim == 1:
            directions_rad = directions * (2 * math.pi)
            dir_diff = directions_rad.unsqueeze(2) - directions_rad.unsqueeze(1) # Shape [B, S, S, 1]
            directional_bias = torch.cos(dir_diff)
            # <<< DÜZELTME: Fazladan son boyutu kaldır >>>
            directional_bias = directional_bias.squeeze(-1) # Shape şimdi [B, S, S] olmalı

        elif self.direction_dim == 2:
            directions_norm = F.normalize(directions, p=2, dim=-1, eps=1e-6)
            directional_bias = torch.bmm(directions_norm, directions_norm.transpose(1, 2)) # Shape [B, S, S]
        else:
            print(f"Uyarı: Yön boyutu {self.direction_dim} bias hesaplaması için desteklenmiyor. Sıfır döndürülüyor.")
            directional_bias = torch.zeros((B, S, S), device=directions.device, dtype=directions.dtype)

        return directional_bias

        # İsteğe bağlı: Öğrenilebilir ölçeklendirme faktörü eklenebilir
        # örn. self.dir_bias_scale = nn.Parameter(torch.tensor(1.0)) __init__ içinde
        # directional_bias = directional_bias * self.dir_bias_scale

        return directional_bias


    def forward(self,
                coords: torch.Tensor,
                class_tokens: torch.Tensor,
                directions: torch.Tensor, # Expects [B, S, direction_dim]
                key_padding_mask: Optional[torch.Tensor] = None # True indicates padding
                ) -> torch.Tensor:
        """
        Forward pass of the HierarchicalFormationTransformer with enhanced bias.

        Args:
            coords (torch.Tensor): Entity coordinates [B, S, coord_dim]. Float.
            class_tokens (torch.Tensor): Entity class indices [B, S]. Long.
            directions (torch.Tensor): Entity directions [B, S, direction_dim]. Float.
                                      *** Ensure this matches direction_dim in __init__! ***
            key_padding_mask (Optional[torch.Tensor]): Mask for padding [B, S]. True indicates padding.

        Returns:
            torch.Tensor: Output logits for formation classes [B, num_formations].
        """
        B, S, Cdim = coords.shape
        assert directions.shape == (B, S, self.direction_dim), \
            f"Directions shape mismatch in forward: expected {(B, S, self.direction_dim)}, got {directions.shape}"
        assert class_tokens.shape == (B, S), \
            f"Class tokens shape mismatch: expected {(B, S)}, got {class_tokens.shape}"
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, S), \
                f"Padding mask shape mismatch: expected {(B, S)}, got {key_padding_mask.shape}"

        device = coords.device

        # --- 1. Compute Input Embeddings ---
        coord_emb = F.gelu(self.coord_fc(coords))           # [B, S, dim_coord]
        direction_emb = F.gelu(self.direction_fc(directions)) # [B, S, dim_dir]
        class_emb = self.class_embedding(class_tokens)        # [B, S, class_embed_dim]
        class_emb = F.gelu(self.class_fc(class_emb))          # [B, S, dim_class]

        # Center coordinates for relative position calculation
        if key_padding_mask is not None:
            keep_mask = ~key_padding_mask.unsqueeze(-1) # [B, S, 1], True means keep
            # Prevent division by zero if all items are padded in a batch element
            num_non_padded = keep_mask.sum(dim=1, keepdim=True).clamp(min=1e-6) # [B, 1, 1]
            center = (coords * keep_mask).sum(dim=1, keepdim=True) / num_non_padded # [B, 1, Cdim]
            # Handle padded elements in relative coords calculation
            rel_coords = (coords - center) * keep_mask # Zero out padded elements
        else:
            center = coords.mean(dim=1, keepdim=True)      # [B, 1, Cdim]
            rel_coords = coords - center                   # [B, S, Cdim]

        rel_emb = F.gelu(self.rel_fc(rel_coords))          # [B, S, dim_rel]

        # --- 2. Fuse Embeddings ---
        fused_emb = torch.cat([coord_emb, direction_emb, class_emb, rel_emb], dim=-1) # [B, S, initial_dim]

        # --- 3. Add Positional Encoding ---
        # x = self.pos_encoder(fused_emb) # [B, S, initial_dim] kaldırılıd
        x = fused_emb      

        # --- 4. Compute Graphormer Biases ---
        spatial_bias = self.compute_spatial_bias(coords)          # [B, S, S]
        # Compute directional bias using the input directions tensor
        directional_bias = self.compute_directional_bias(directions) # [B, S, S]

        # --- 5. Pass through Hierarchical Transformer Stages ---
        # Pass both computed biases to the encoder stack
        current_input = x
        for i, (proj, stage) in enumerate(zip(self.projections, self.stages)):
            projected_input = proj(current_input) # Apply projection (Identity for first stage)
            output = stage(projected_input,
                           spatial_bias=spatial_bias,
                           directional_bias=directional_bias,
                           src_key_padding_mask=key_padding_mask) # Pass padding mask
            current_input = output # Output of stage becomes input for next projection

        # --- 6. Pooling and Final Classification ---
        # current_input shape is now [B, S, last_stage_dim]

        # Calculate attention weights for pooling
        pool_logits = self.pooling_weights_fc(current_input) # [B, S, 1]

        # Apply masking to pooling logits before softmax
        if key_padding_mask is not None:
            pool_logits = pool_logits.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))

        pool_attn_weights = F.softmax(pool_logits, dim=1) # [B, S, 1]

        # Apply weighted sum pooling: (B, S, E) * (B, S, 1) -> sum over S -> (B, E)
        # Use masked output from last stage for pooling calculation
        # Masking ensures padded elements don't contribute (their weight is ~0 from softmax)
        pooled_output = (current_input * pool_attn_weights).sum(dim=1) # [B, E]

        # Apply final normalization and classification layer
        pooled_output_norm = self.output_norm(pooled_output) # [B, E]
        out_logits = self.output_fc(pooled_output_norm)     # [B, num_formations]

        return out_logits

