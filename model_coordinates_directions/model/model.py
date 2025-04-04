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
        # Formula ensures frequencies decrease along the dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Create matrix for positional encodings: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (handle cases where d_model is odd)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2]) # Match shape if d_model is odd

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer, not a model parameter (no gradients needed)
        self.register_buffer('pe', pe, persistent=False) # persistent=False avoids saving in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor with added positional encodings and dropout,
                          shape [batch_size, seq_len, d_model].
        """
        # x shape: [batch_size, seq_len, d_model]
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

# --- Graphormer-style Attention ---

class GraphormerMultiheadAttention(nn.Module):
    """
    Multi-head self-attention with Graphormer-style edge bias.

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

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor [B, T, E]. (T = target seq len)
            key (torch.Tensor): Key tensor [B, S, E]. (S = source seq len)
            value (torch.Tensor): Value tensor [B, S, E].
            edge_bias (Optional[torch.Tensor]): Graphormer edge bias [B, T, S]. Added to
                                                attention scores before softmax.
            key_padding_mask (Optional[torch.Tensor]): Mask for key padding [B, S].
                                                      Positions with True are ignored.
            attn_mask (Optional[torch.Tensor]): Additional attention mask [T, S] or [B, T, S].
                                                Added to attention scores.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Attention output [B, T, E].
                - Attention weights [B, H, T, S] (optional, usually None during training).
                  Note: Weights are not returned in this implementation for simplicity.
        """
        B, T, E = query.shape
        B, S, E_k = key.shape
        _B, _S, E_v = value.shape # Value seq len must match key seq len
        assert E == E_k == E_v, "Embedding dimensions must match"
        assert B == _B == _B, "Batch sizes must match"
        assert S == _S, "Key and Value sequence lengths must match"

        # 1. Linear projections + split into heads
        # (B, Seq, E) -> (B, Seq, H, D) -> (B, H, Seq, D)
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled dot-product attention
        # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Apply masks and biases BEFORE softmax
        if attn_mask is not None:
            # Needs correct broadcasting: e.g., [T, S] -> [1, 1, T, S]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: # [B, T, S] -> [B, 1, T, S]
                 attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask # Add mask biases (e.g., -inf for masked positions)

        if key_padding_mask is not None:
            # key_padding_mask [B, S] -> [B, 1, 1, S] to broadcast over heads and query positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # --- Add Graphormer Edge Bias ---
        if edge_bias is not None:
            # edge_bias shape: [B, T, S] -> [B, 1, T, S] for broadcasting over heads
            # Ensure T == S for self-attention edge bias
            if T == S:
                 attn_scores = attn_scores + edge_bias.unsqueeze(1)
            else:
                 # Handle cross-attention case if needed, or raise error
                 # For typical encoder self-attention, T == S
                 pass # Ignore edge_bias if T != S in this simple version

        # 4. Softmax and Dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 5. Weighted sum of values
        # (B, H, T, S) @ (B, H, S, D) -> (B, H, T, D)
        attn_output = attn_probs @ v

        # 6. Concatenate heads and final linear projection
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        output = self.out_proj(attn_output)

        # Return output, and optionally attention weights (set to None here)
        return output, None # Returning None for weights for simplicity

# --- Custom Transformer Layer and Encoder ---

class GraphormerTransformerLayer(nn.Module):
    """
    Transformer Encoder Layer with Graphormer Attention and optional Adapter.
    Uses pre-layer normalization (Pre-LN).

    Args:
        d_model (int): Input dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        activation (nn.Module): Activation function for FFN. Default: nn.GELU.
        use_adapter (bool): Whether to include an adapter block. Default: False.
        adapter_dim (int): Bottleneck dimension for the adapter. Default: 64.
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
                edge_bias: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None, # Optional standard attention mask
                src_key_padding_mask: Optional[torch.Tensor] = None # Padding mask
                ) -> torch.Tensor:
        """
        Forward pass for the Transformer layer (Pre-LN structure).

        Args:
            src (torch.Tensor): Input tensor [B, S, E].
            edge_bias (Optional[torch.Tensor]): Graphormer edge bias [B, S, S].
            src_mask (Optional[torch.Tensor]): Standard attention mask [S, S] or [B, S, S].
            src_key_padding_mask (Optional[torch.Tensor]): Padding mask [B, S].

        Returns:
            torch.Tensor: Output tensor [B, S, E].
        """
        # --- Self-Attention Part (Pre-LN) ---
        residual = src
        x = self.norm1(src)
        # Pass edge_bias and key_padding_mask to custom attention
        attn_output, _ = self.self_attn(x, x, x, # Self-attention query=key=value
                                        edge_bias=edge_bias,
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
    A stack of Transformer Encoder Layers that correctly passes edge_bias.

    Args:
        encoder_layer (nn.Module): An instance of GraphormerTransformerLayer.
        num_layers (int): Number of layers in the encoder.
        norm (Optional[nn.Module]): Optional layer normalization after the stack.
    """
    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        # Create deep copies of the encoder layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, # Standard attn mask (e.g., causal)
                src_key_padding_mask: Optional[torch.Tensor] = None # Padding mask
                ) -> torch.Tensor:
        """
        Passes input through the stack of layers.

        Args:
            src (torch.Tensor): Input tensor [B, S, E].
            edge_bias (Optional[torch.Tensor]): Graphormer edge bias [B, S, S].
            mask (Optional[torch.Tensor]): Standard attention mask passed to all layers.
            src_key_padding_mask (Optional[torch.Tensor]): Padding mask passed to all layers.

        Returns:
            torch.Tensor: Output tensor [B, S, E].
        """
        output = src
        for layer in self.layers:
            output = layer(output,
                           edge_bias=edge_bias,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# --- Main Hierarchical Model ---

class HierarchicalFormationTransformer(nn.Module):
    """
    Hierarchical Transformer for formation classification using coordinates,
    class types, and directions, with Graphormer-style spatial bias.

    Args:
        num_formations (int): Number of output formation classes.
        class_vocab_size (int): Number of unique entity classes (e.g., tank, infantry).
        class_embed_dim (int): Initial dimension for class embeddings lookup.
        coord_dim (int): Dimension of input coordinates (usually 2 for 2D). Default: 2.
        direction_dim (int): Dimension of input directions (usually 1). Default: 1.
        stage_dims (List[int]): List of embedding dimensions for each transformer stage.
                                Example: [512, 256, 128].
        num_heads (int): Number of attention heads in each stage.
        num_layers (List[int]): List of number of layers for each transformer stage.
                                Must have the same length as stage_dims.
        dropout_stages (List[float]): List of dropout rates for each stage.
                                      Must have the same length as stage_dims.
        use_adapter (bool): Whether to use Adapter modules in transformer layers.
        adapter_dim (int): Hidden dimension for Adapter modules.
        pos_type (str): Type of positional encoding ('learnable' or 'sinusoidal').
        max_len (int): Maximum sequence length for positional encodings.
        ffn_ratio (int): Ratio to calculate dim_feedforward (dim_feedforward = ffn_ratio * d_model).
                         Default: 4.
    """
    def __init__(self,
                 num_formations: int,
                 class_vocab_size: int,
                 class_embed_dim: int = 64,
                 coord_dim: int = 2,
                 direction_dim: int = 1,
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

        num_stages = len(stage_dims)
        initial_dim = stage_dims[0] # Dimension after initial embedding fusion

        # --- Define dimensions for concatenated input embeddings ---
        # Allocate space within initial_dim for each feature type. Adjust ratios if needed.
        total_parts = 4 # coord, dir, class, rel_coord
        base_dim = initial_dim // total_parts
        remainder = initial_dim % total_parts
        # Distribute dimensions (add remainder to the first one, e.g., coord)
        self.dim_coord = base_dim + remainder
        self.dim_dir = base_dim
        self.dim_class = base_dim
        self.dim_rel = base_dim
        assert self.dim_coord + self.dim_dir + self.dim_class + self.dim_rel == initial_dim, \
               "Input dimension allocation calculation error."

        # --- Input Projection Layers ---
        self.coord_fc = nn.Linear(coord_dim, self.dim_coord)
        self.direction_fc = nn.Linear(direction_dim, self.dim_dir)
        self.class_embedding = nn.Embedding(class_vocab_size, class_embed_dim)
        self.class_fc = nn.Linear(class_embed_dim, self.dim_class)
        self.rel_fc = nn.Linear(coord_dim, self.dim_rel) # Relative coords are still 2D

        # --- Positional Encoding ---
        if pos_type == 'learnable':
            self.pos_encoder = LearnablePositionalEmbedding(max_len=max_len, d_model=initial_dim)
        else: # sinusoidal
            # Use dropout from the first stage for sinusoidal encoding
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
                    nn.GELU(), # Use GELU consistently
                    nn.Dropout(stage_dropout)
                )
                self.projections.append(proj)
            else:
                self.projections.append(nn.Identity()) # No projection before first stage

            # Transformer Encoder for the current stage
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

        # Weighted pooling layer (Attention Pooling)
        # Projects each entity's final representation to a single logit for attention weight
        self.pooling_weights_fc = nn.Linear(last_stage_dim, 1)

        # Normalization after the last stage and pooling
        self.output_norm = nn.LayerNorm(last_stage_dim)
        # Final classification layer
        self.output_fc = nn.Linear(last_stage_dim, num_formations)

    def compute_edge_bias(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes Graphormer-style edge bias based on pairwise distances.

        Args:
            coords (torch.Tensor): Coordinates tensor [B, S, coord_dim].

        Returns:
            torch.Tensor: Edge bias tensor [B, S, S]. Closer points get higher
                          (less negative) bias. Returns zeros if S <= 1.
        """
        B, S, _ = coords.shape
        if S <= 1:
             # Cannot compute pairwise distance for single element sequences
             return torch.zeros((B, S, S), device=coords.device, dtype=coords.dtype)

        # Calculate pairwise Euclidean distances: [B, S, S]
        dist = torch.cdist(coords, coords, p=2)

        # Normalize distances (optional, helps stabilize). Add epsilon for safety.
        # Normalizing by mean distance in the formation.
        norm_factor = dist.mean(dim=(1, 2), keepdim=True)
        # Avoid division by zero if all points are coincident (mean distance is 0)
        edge_bias = -dist / (norm_factor + 1e-6)

        # Alternative: Simple scaling (might need tuning)
        # scale_factor = -1.0
        # edge_bias = dist * scale_factor

        return edge_bias

    def forward(self,
                coords: torch.Tensor,
                class_tokens: torch.Tensor,
                directions: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass of the HierarchicalFormationTransformer.

        Args:
            coords (torch.Tensor): Entity coordinates [B, S, coord_dim]. Float.
            class_tokens (torch.Tensor): Entity class indices [B, S]. Long.
            directions (torch.Tensor): Entity directions [B, S, direction_dim]. Float.
            key_padding_mask (Optional[torch.Tensor]): Mask for padding [B, S].
                                                      True indicates a padded position.

        Returns:
            torch.Tensor: Output logits for formation classes [B, num_formations].
        """
        B, S, _ = coords.shape
        device = coords.device

        # --- 1. Compute Input Embeddings ---
        coord_emb = F.gelu(self.coord_fc(coords))           # [B, S, dim_coord]
        direction_emb = F.gelu(self.direction_fc(directions)) # [B, S, dim_dir]
        class_emb = self.class_embedding(class_tokens)        # [B, S, class_embed_dim]
        class_emb = F.gelu(self.class_fc(class_emb))          # [B, S, dim_class]

        # Center coordinates for relative position calculation
        # Handle padding by masking before mean calculation (if mask provided)
        if key_padding_mask is not None:
            # Invert mask for calculation (True means keep)
            keep_mask = ~key_padding_mask.unsqueeze(-1) # [B, S, 1]
            coords_masked = coords * keep_mask
            num_non_padded = keep_mask.sum(dim=1, keepdim=True) # [B, 1, 1]
            center = coords_masked.sum(dim=1, keepdim=True) / (num_non_padded + 1e-6) # [B, 1, coord_dim]
        else:
            center = coords.mean(dim=1, keepdim=True)      # [B, 1, coord_dim]

        rel_coords = coords - center                       # [B, S, coord_dim]
        rel_emb = F.gelu(self.rel_fc(rel_coords))          # [B, S, dim_rel]

        # --- 2. Fuse Embeddings ---
        # Concatenate all four projected embeddings
        fused_emb = torch.cat([coord_emb, direction_emb, class_emb, rel_emb], dim=-1) # [B, S, initial_dim]

        # --- 3. Add Positional Encoding ---
        x = self.pos_encoder(fused_emb) # [B, S, initial_dim]

        # --- 4. Compute Edge Bias (Spatial Information) ---
        edge_bias = self.compute_edge_bias(coords) # [B, S, S]

        # --- 5. Pass through Hierarchical Transformer Stages ---
        for i, (proj, stage) in enumerate(zip(self.projections, self.stages)):
            # Apply projection (Identity for the first stage)
            x = proj(x) # Shape changes after proj if i > 0
            # Pass through transformer stage, providing edge_bias and padding mask
            x = stage(x,
                      edge_bias=edge_bias, # Provide edge bias computed once
                      src_key_padding_mask=key_padding_mask) # Provide padding mask

        # --- 6. Pooling and Final Classification ---
        # x shape is now [B, S, last_stage_dim]

        # Calculate attention weights for pooling
        # pool_logits shape: [B, S, 1]
        pool_logits = self.pooling_weights_fc(x)

        # Apply masking to pooling logits before softmax
        if key_padding_mask is not None:
            # Masked positions should have -inf logit so their weight becomes 0
            pool_logits = pool_logits.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))

        # Calculate pooling weights: [B, S, 1]
        pool_attn_weights = F.softmax(pool_logits, dim=1)

        # Apply weighted sum pooling: [B, S, E] * [B, S, 1] -> sum -> [B, E]
        pooled_output = (x * pool_attn_weights).sum(dim=1)

        # Apply final normalization and classification layer
        pooled_output_norm = self.output_norm(pooled_output) # [B, E]
        out_logits = self.output_fc(pooled_output_norm)     # [B, num_formations]

        return out_logits

