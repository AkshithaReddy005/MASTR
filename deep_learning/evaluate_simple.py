"""
Simple Evaluation Script with Matching Model Architecture
"""
import torch
import torch.nn as nn
import numpy as np
import argparse

class SimpleMAAM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, customer_features, vehicle_state, mask=None):
        # Encode customers
        encoded = self.encoder(customer_features)
        
        # Simple attention
        q = self.query(vehicle_state)
        k = self.key(encoded)
        v = self.value(encoded)
        
        # Attention scores
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        return scores
    
    def sample_action(self, customer_features, vehicle_state, mask=None, greedy=False):
        scores = self.forward(customer_features, vehicle_state, mask)
        probs = torch.softmax(scores, dim=-1)
        
        if greedy:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
            
        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
        return action, log_prob

def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate Simple MAAM Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--render', action='store_true', help='Render environment')
    args = parser.parse_args()
    
    # Create model
    model = SimpleMAAM()
    
    # Load only the model weights (ignore optimizer state)
    checkpoint = torch.load(args.model, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Simple test
    print("Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    test_customers = torch.randn(1, 10, 8)  # batch_size, num_customers, features
    test_vehicle = torch.randn(1, 8)        # batch_size, features
    test_mask = torch.zeros(1, 10).bool()   # batch_size, num_customers
    
    with torch.no_grad():
        action, _ = model.sample_action(test_customers, test_vehicle, test_mask, greedy=True)
        print(f"Test action: {action.item()}")
    
    print("\nTo run full evaluation, use the environment code from train_final.py")
    print("This was just a quick test to verify model loading works.")

if __name__ == "__main__":
    evaluate()
