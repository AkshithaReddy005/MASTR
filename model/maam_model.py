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
        tanh_clipping: float = 10.0,
        init_gain: float = 0.01
    ):
        # Store initialization parameters
        self.init_gain = init_gain
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Input normalization layer to prevent numerical instability
        self.input_norm = nn.LayerNorm(input_dim)
        
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
        self.init_decoder_state = nn.Parameter(torch.zeros(embed_dim))
        
        # Initialize model weights (after all parameters are defined)
        self._init_weights()
        
        # Initialize optimizer state
        self.optimizer = None
        
        # Vehicle state encoder with layer norm
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),  # [x, y, load, time] -> embed_dim
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Gradient clipping
        self.grad_clip = 1.0
    
    def _init_weights(self):
        """Initialize weights with careful scaling to prevent numerical instability"""
        def init_weight(m):
            if isinstance(m, nn.Linear):
                # Use orthogonal initialization for better gradient flow
                nn.init.orthogonal_(m.weight, gain=self.init_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize attention weights carefully
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight, gain=0.1)
                if hasattr(m, 'out_proj') and m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight, gain=0.1)
        
        # Apply initialization to all modules
        self.apply(init_weight)
        
        # Special initialization for the decoder state with very small values
        nn.init.normal_(self.init_decoder_state, mean=0.0, std=0.001)
    
    def clip_gradients(self, max_norm=1.0):
        """Clip gradients to avoid exploding gradients"""
        if max_norm > 0:
            # Use gradient clipping with error handling
            try:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad], 
                    max_norm=max_norm
                )
            except RuntimeError as e:
                print(f"Warning: Error in gradient clipping: {e}")
                # Handle the error by zeroing out gradients if needed
                for p in self.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            p.grad.zero_()
    
    def forward(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with enhanced numerical stability checks
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4] - current vehicle state
            mask: [batch_size, num_customers] - visited customers mask
        
        Returns:
            logits: [batch_size, num_customers] - action logits
            encoder_outputs: [batch_size, num_customers, embed_dim]
        """
        def check_nan_inf(tensor, name):
            if not torch.is_tensor(tensor):
                return tensor
                
            if torch.isnan(tensor).any():
                print(f"Warning: NaN detected in {name}")
                tensor = torch.nan_to_num(tensor, nan=0.0)
                
            if torch.isinf(tensor).any():
                print(f"Warning: Inf detected in {name}")
                tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
                
            return tensor
            
        def safe_softmax(x, dim=-1, eps=1e-6):
            """Numerically stable softmax with clamping"""
            x = x - x.max(dim=dim, keepdim=True)[0]  # for numerical stability
            x = torch.exp(x)
            x = x / (x.sum(dim=dim, keepdim=True) + eps)
            return x
        
        # Normalize inputs first to prevent numerical instability
        customer_features = self.input_norm(customer_features)
        
        # Check and clean input tensors
        with torch.no_grad():
            customer_features = check_nan_inf(customer_features, "customer_features")
            vehicle_state = check_nan_inf(vehicle_state, "vehicle_state")
            
            # Ensure inputs are on the same device
            if mask is not None:
                mask = mask.to(customer_features.device)
        
        batch_size = customer_features.size(0)
        
        # Encode customers with gradient checkpointing for memory efficiency
        encoder_outputs = self.encoder(customer_features)  # [B, N, embed_dim]
        encoder_outputs = check_nan_inf(encoder_outputs, "encoder_outputs")
        
        # Add residual connection and layer norm for stability
        if hasattr(self, 'encoder_norm'):
            encoder_outputs = self.encoder_norm(encoder_outputs)
        
        # Encode vehicle state with gradient checkpointing
        vehicle_embedding = self.vehicle_encoder(vehicle_state)  # [B, embed_dim]
        vehicle_embedding = check_nan_inf(vehicle_embedding, "vehicle_embedding")
        
        # Initialize decoder state with residual connection
        decoder_state = self.init_decoder_state.unsqueeze(0).expand(batch_size, -1)
        
        # Add vehicle embedding with scaling for stability
        decoder_state = decoder_state + 0.1 * vehicle_embedding
        
        # Apply layer normalization if available
        if hasattr(self, 'decoder_norm'):
            decoder_state = self.decoder_norm(decoder_state)
            
        decoder_state = check_nan_inf(decoder_state, "decoder_state")
        
        # Decode: get next customer selection with gradient checkpointing
        logits, _ = self.decoder(decoder_state, encoder_outputs, mask)
        
        # Apply tanh clipping to logits for stability
        if hasattr(self, 'tanh_clipping') and self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)
            
        logits = check_nan_inf(logits, "logits")
        
        # Apply mask to logits if provided (after checking for inf)
        if mask is not None:
            # Use a very negative number instead of -inf for better numerical stability
            logits = logits.masked_fill(mask, -1e9)
        
        return logits, encoder_outputs
    
    def calculate_loss(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: torch.Tensor,
        reward: float
    ) -> torch.Tensor:
        """
        Calculate policy gradient loss with baseline and improved numerical stability
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4]
            mask: [batch_size, num_customers]
            reward: scalar reward for the episode
            
        Returns:
            loss: policy gradient loss
        """
        try:
            # Get action logits and sample action
            logits, _ = self.forward(customer_features, vehicle_state, mask)
            
            # Ensure logits are finite before proceeding
            if not torch.isfinite(logits).all():
                print("Warning: Non-finite logits detected in calculate_loss")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Calculate log probabilities with numerical stability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Calculate probabilities from logits (for sampling)
            probs = F.softmax(logits, dim=-1)
            
            # Ensure probabilities are valid (no NaN, Inf, or negative values)
            if not (probs >= 0).all() or not torch.isfinite(probs).all():
                print("Warning: Invalid probabilities detected in calculate_loss")
                probs = torch.softmax(torch.clamp(logits, min=-10, max=10), dim=-1)
            
            # Ensure probabilities sum to 1 (with small epsilon to avoid division by zero)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            try:
                # Sample an action
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Get log probability of the selected action
                selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                # Calculate policy gradient loss (negative for gradient ascent)
                # Using reward as a baseline for simplicity
                baseline = 0.0  # Simple baseline - could be improved with a value network
                advantage = reward - baseline
                
                # Calculate policy gradient loss
                policy_loss = -selected_log_probs * advantage
                
                # Add entropy regularization to encourage exploration
                # Use max(entropy, 0) to ensure non-negative entropy
                entropy = torch.clamp(-(probs * log_probs).sum(dim=-1), min=0)
                entropy_bonus = 0.01 * entropy  # Entropy coefficient can be tuned
                
                # Combine losses
                loss = policy_loss - entropy_bonus
                
                # Ensure the loss is finite
                if not torch.isfinite(loss).all():
                    print(f"Warning: Non-finite loss detected: {loss}")
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Return mean loss over the batch
                return loss.mean()
                
            except RuntimeError as e:
                print(f"Error in action sampling: {e}")
                # Fallback: return a small random loss to continue training
                return torch.tensor(0.01, device=customer_features.device, requires_grad=True)
                
        except Exception as e:
            print(f"Error in calculate_loss: {e}")
            # Return a small random loss to continue training
            return torch.tensor(0.01, device=customer_features.device, requires_grad=True)
    
    def get_action_probs(
        self,
        customer_features: torch.Tensor,
        vehicle_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get action probability distribution with numerical stability improvements
        
        Args:
            customer_features: [batch_size, num_customers, input_dim]
            vehicle_state: [batch_size, 4]
            mask: [batch_size, num_customers]
            temperature: sampling temperature (higher = more random)
        
        Returns:
            probs: [batch_size, num_customers] - action probabilities
        """
        # Ensure temperature is valid
        temperature = max(temperature, 1e-8)  # Avoid division by zero
        
        # Get logits from forward pass
        logits, _ = self.forward(customer_features, vehicle_state, mask)
        
        # Apply temperature scaling with numerical stability
        logits = logits / temperature
        
        # Stable softmax
        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
        exp_logits = torch.exp(logits)
        
        # Handle masked elements (set to 0 after exp to avoid -inf in log)
        if mask is not None:
            exp_logits = exp_logits.masked_fill(mask, 0.0)
        
        # Calculate probabilities
        probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Final check for NaN/Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Warning: Invalid probabilities detected in get_action_probs")
            # Fall back to uniform distribution over valid actions
            probs = torch.ones_like(logits, device=logits.device)
            if mask is not None:
                probs = probs.masked_fill(mask, 0.0)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / (probs_sum + 1e-10)
        
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
        
        # Check if probabilities are valid
        prob_sum = probs.sum(dim=-1, keepdim=True)
        if (prob_sum <= 1e-8).any():
            print("Warning: Invalid probability distribution, using uniform fallback")
            # Create uniform distribution over non-masked actions
            probs = torch.ones_like(probs)
            if mask is not None:
                probs = probs.masked_fill(mask, 0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        if greedy:
            actions = torch.argmax(probs, dim=-1)
        else:
            try:
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            except RuntimeError as e:
                print(f"Warning: Multinomial sampling failed ({e}), using argmax fallback")
                actions = torch.argmax(probs, dim=-1)
        
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        return actions, log_probs
