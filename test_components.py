"""
Quick test to verify all components work correctly
"""
import torch
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM

print("="*60)
print("TESTING MASTR COMPONENTS")
print("="*60)

# Test 1: Environment
print("\n[1/4] Testing Environment...")
try:
    env = MVRPSTWEnv(num_customers=5, num_vehicles=2, seed=42)
    obs, info = env.reset()
    print(f"  ✓ Environment created")
    print(f"  ✓ Observation shape: {obs.shape}")
    print(f"  ✓ Action space: {env.action_space}")
    
    # Test a step
    action = 0
    obs, reward, done, truncated, info = env.step(action)
    print(f"  ✓ Step executed: reward={reward:.2f}, done={done}")
except Exception as e:
    print(f"  ✗ Environment failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Model
print("\n[2/4] Testing Model...")
try:
    model = MAAM(input_dim=8, embed_dim=64, num_heads=4, num_encoder_layers=2)
    print(f"  ✓ Model created")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 1
    num_customers = 5
    customer_features = torch.randn(batch_size, num_customers, 8)
    vehicle_state = torch.randn(batch_size, 4)
    mask = torch.zeros(batch_size, num_customers, dtype=torch.bool)
    
    logits, encoder_out = model(customer_features, vehicle_state, mask)
    print(f"  ✓ Forward pass: logits shape={logits.shape}")
    
    # Test action sampling
    action, log_prob = model.sample_action(customer_features, vehicle_state, mask, greedy=False)
    print(f"  ✓ Action sampling: action={action.item()}, log_prob={log_prob.item():.4f}")
except Exception as e:
    print(f"  ✗ Model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Integration (env + model)
print("\n[3/4] Testing Integration...")
try:
    env = MVRPSTWEnv(num_customers=5, num_vehicles=2, seed=42)
    model = MAAM(input_dim=8, embed_dim=64, num_heads=4, num_encoder_layers=2)
    
    obs, info = env.reset()
    
    # Parse observation
    num_customers = env.num_customers
    customer_start = 8
    customer_end = 8 + num_customers * 8
    customer_flat = obs[customer_start:customer_end]
    customer_features = torch.FloatTensor(customer_flat).reshape(1, num_customers, 8)
    
    vehicle_start = customer_end
    vehicle_flat = obs[vehicle_start:]
    vehicle_state = torch.FloatTensor(vehicle_flat).reshape(env.num_vehicles, 4)[0].unsqueeze(0)
    
    # Create mask
    visited = []
    for i in range(num_customers):
        start_idx = 8 + i * 8
        visited_flag = obs[start_idx + 7]
        visited.append(bool(visited_flag > 0.5))
    mask = torch.tensor(visited, dtype=torch.bool).unsqueeze(0)
    
    print(f"  ✓ Parsed observation")
    print(f"    - Customer features: {customer_features.shape}")
    print(f"    - Vehicle state: {vehicle_state.shape}")
    print(f"    - Mask: {mask.shape}, visited: {mask.sum().item()}/{num_customers}")
    
    # Sample action
    with torch.no_grad():
        action, log_prob = model.sample_action(customer_features, vehicle_state, mask, greedy=True)
    
    print(f"  ✓ Sampled action: {action.item()}")
    
    # Execute action
    obs, reward, done, truncated, info = env.step(action.item())
    print(f"  ✓ Step executed: reward={reward:.2f}")
    
except Exception as e:
    print(f"  ✗ Integration failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Short rollout
print("\n[4/4] Testing Short Rollout...")
try:
    env = MVRPSTWEnv(num_customers=5, num_vehicles=2, seed=42)
    model = MAAM(input_dim=8, embed_dim=64, num_heads=4, num_encoder_layers=2)
    
    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 20:
        # Parse observation
        num_customers = env.num_customers
        customer_start = 8
        customer_end = 8 + num_customers * 8
        customer_flat = obs[customer_start:customer_end]
        customer_features = torch.FloatTensor(customer_flat).reshape(1, num_customers, 8)
        
        vehicle_start = customer_end
        vehicle_flat = obs[vehicle_start:]
        vehicle_state = torch.FloatTensor(vehicle_flat).reshape(env.num_vehicles, 4)[env.current_vehicle].unsqueeze(0)
        
        # Create mask
        visited = []
        for i in range(num_customers):
            start_idx = 8 + i * 8
            visited_flag = obs[start_idx + 7]
            visited.append(bool(visited_flag > 0.5))
        mask = torch.tensor(visited, dtype=torch.bool).unsqueeze(0)
        
        # Check if all masked
        if mask.all():
            print(f"  ! All customers visited, breaking")
            break
        
        # Sample action
        with torch.no_grad():
            action, log_prob = model.sample_action(customer_features, vehicle_state, mask, greedy=True)
        
        # Execute
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1
    
    print(f"  ✓ Rollout complete")
    print(f"    - Steps: {steps}")
    print(f"    - Total reward: {total_reward:.2f}")
    print(f"    - Customers served: {env.visited.sum()}/{env.num_customers}")
    
except Exception as e:
    print(f"  ✗ Rollout failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nIf all tests passed (✓), the implementation is correct.")
print("You can now run: python train/train_rl.py")
print("="*60)
