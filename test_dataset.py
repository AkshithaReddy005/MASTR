"""
Quick test to verify dataset loading
"""
import os
from env.mvrp_env import MVRPSTWEnv

print("="*60)
print("DATASET LOADING TEST")
print("="*60)

# Check if CSV file exists
data_path = "data/raw/VRP.csv"
file_exists = os.path.exists(data_path)

print(f"\n1. Checking for CSV file...")
print(f"   Path: {data_path}")
print(f"   Exists: {'✓ YES' if file_exists else '✗ NO'}")

if not file_exists:
    print("\n⚠ Dataset not found!")
    print("   Please download 'VRP - C101.csv' from Kaggle and place it in data/raw/")
    print("   See DATASET_SETUP.md for instructions")
    print("\n   For now, testing with random data...")
    data_path = None

# Create environment
print(f"\n2. Creating environment...")
env = MVRPSTWEnv(
    num_customers=10,
    num_vehicles=2,
    data_path=data_path
)

# Reset to trigger data loading
print(f"\n3. Loading data...")
obs, info = env.reset()

print(f"\n4. Data loaded successfully!")
print(f"   - Number of customers: {env.num_customers}")
print(f"   - Depot location: {env.depot_loc}")
print(f"   - Using real data: {'✓ YES' if env.use_real_data else '✗ NO (random)'}")

if env.use_real_data:
    print(f"\n5. Sample customer data:")
    for i in range(min(3, env.num_customers)):
        print(f"   Customer {i+1}:")
        print(f"     Location: ({env.customer_locations[i][0]:.1f}, {env.customer_locations[i][1]:.1f})")
        print(f"     Demand: {env.demands[i]:.1f}")
        print(f"     Time Window: [{env.time_windows[i][0]:.0f}, {env.time_windows[i][1]:.0f}]")

print("\n" + "="*60)
if env.use_real_data:
    print("✓ SUCCESS: Real dataset loaded correctly!")
else:
    print("⚠ Using random data (download dataset to use real data)")
print("="*60)
