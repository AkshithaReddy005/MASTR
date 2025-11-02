"""
Multi-Agent Attention Model (MAAM)
Transformer-based encoder-decoder architecture for MVRPSTW
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for customer embeddings
    Encodes customer features (location, demand, time windows) into context vectors
    """
    def __init__(
        self,
        input_dim: int = 8,  # [x, y, demand, start_time, end_time, penalty_early, penalty_late, visited]
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, customer_features):
        """
        Args:
            customer_features: [batch_size, num_customers, input_dim]
        Returns:
            encoded: [batch_size, num_customers, embed_dim]
        """
        # Project to embedding dimension
        x = self.input_projection(customer_features)  # [B, N, embed_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # [B, N, embed_dim]
        encoded = self.layer_norm(encoded)
        
        return encoded


class PointerDecoder(nn.Module):
    """
    Pointer Network decoder with attention
    Selects next customer based on current context
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        tanh_clipping: float = 10.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        
        # Query projection (from decoder state)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Key and value projections (from encoder outputs)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Context projection
        self.context_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Pointer mechanism
        self.pointer_query = nn.Linear(embed_dim, embed_dim)
        self.pointer_key = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: [batch_size, embed_dim] - current decoder state
            encoder_outputs: [batch_size, num_customers, embed_dim] - encoded customer features
            mask: [batch_size, num_customers] - mask for visited customers (True = mask out)
        
        Returns:
            logits: [batch_size, num_customers] - selection probabilities
            context: [batch_size, embed_dim] - attended context vector
        """
        batch_size, num_customers, _ = encoder_outputs.shape
        
        # Expand decoder state for attention
        query = decoder_state.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Multi-head attention
        context, attn_weights = self.attention(
            query=query,
            key=encoder_outputs,
            value=encoder_outputs,
            key_padding_mask=mask
        )  # context: [B, 1, embed_dim]
        
        context = context.squeeze(1)  # [B, embed_dim]
        
        # Combine decoder state and context
        combined = torch.cat([decoder_state, context], dim=-1)  # [B, embed_dim*2]
        new_state = self.context_proj(combined)  # [B, embed_dim]
        
        # Pointer mechanism: compute compatibility scores
        query_pointer = self.pointer_query(new_state).unsqueeze(1)  # [B, 1, embed_dim]
        keys_pointer = self.pointer_key(encoder_outputs)  # [B, N, embed_dim]
        
        # Compute logits via dot product
        logits = torch.bmm(query_pointer, keys_pointer.transpose(1, 2)).squeeze(1)  # [B, N]
        logits = logits / math.sqrt(self.embed_dim)
        
        # Apply tanh clipping
        logits = self.tanh_clipping * torch.tanh(logits)
        
        # Apply mask (set masked positions to large negative value)
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        
        return logits, new_state


class MAAM(nn.Module):
    """
    Multi-Agent Attention Model
    Complete model combining encoder and decoder for MVRPSTW
    """
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        tanh_clipping: float = 10.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Encoder: encodes all customers
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Decoder: selects next customer
        self.decoder = PointerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            tanh_clipping=tanh_clipping
        )
        
        # Initial decoder state (learnable)
        self.init_decoder_state = nn.Parameter(torch.randn(embed_dim))
        
        # Vehicle state encoder
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),  # [x, y, load, time] -> embed_dim
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4] - current vehicle state
            mask: [batch_size, num_customers] - visited customers mask
        
        Returns:
            logits: [batch_size, num_customers] - action probabilities
            encoder_outputs: [batch_size, num_customers, embed_dim]
        """
        batch_size = customer_features.size(0)
        
        # Encode customers
        encoder_outputs = self.encoder(customer_features)  # [B, N, embed_dim]
        
        # Encode vehicle state
        vehicle_embedding = self.vehicle_encoder(vehicle_state)  # [B, embed_dim]
        
        # Initialize decoder state (combine learnable init + vehicle state)
        decoder_state = self.init_decoder_state.unsqueeze(0).expand(batch_size, -1)
        decoder_state = decoder_state + vehicle_embedding
        
        # Decode: get next customer selection
        logits, _ = self.decoder(decoder_state, encoder_outputs, mask)
        
        return logits, encoder_outputs
    
    def get_action_probs(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get action probability distribution
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4]
            mask: [batch_size, num_customers]
            temperature: sampling temperature
        
        Returns:
            probs: [batch_size, num_customers]
        """
        logits, _ = self.forward(customer_features, vehicle_state, mask)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs
    
    def sample_action(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        greedy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4]
            mask: [batch_size, num_customers]
            greedy: if True, select argmax; else sample
        
        Returns:
            actions: [batch_size] - selected customer indices
            log_probs: [batch_size] - log probabilities of actions
        """
        probs = self.get_action_probs(customer_features, vehicle_state, mask)
        
        if greedy:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        return actions, log_probs
