import torch
from border_env import BorderEnv

# 1. What dim did the checkpoint train on?
ck = torch.load('checkpoints/system_C_seed0/step_1001472.pt', map_location='cpu')
print("Checkpoint obs_dim:", ck['policy_state_dict']['net.0.weight'].shape[1])
print("Config stored:", ck.get('config', {}).get('obs_dim', 'NOT STORED'))

# 2. What dim is the env actually producing right now?
env = BorderEnv(use_pybullet=False, seed=0)
obs, _ = env.reset()
print("Env DRONE_OBS_DIM constant:", env.drone_obs_dim)
print("Actual obs shape:", obs['drone_0'].shape)