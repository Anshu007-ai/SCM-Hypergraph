import torch
import numpy as np

checkpoint = torch.load('outputs/checkpoints/best.pt', map_location='cpu', weights_only=False)

print('=== CHECKPOINT INFO ===')
print('Epoch:', checkpoint.get('epoch', 'N/A'))
print('Keys:', list(checkpoint.keys()))
print('')

print('=== MODEL STATE DICT KEYS ===')
state_dict_keys = list(checkpoint['model_state_dict'].keys())
print(f'Total parameters: {len(state_dict_keys)}')
print('First 10 keys:')
for k in state_dict_keys[:10]:
    print(f'  {k}')
print('Last 5 keys:')
for k in state_dict_keys[-5:]:
    print(f'  {k}')
print('')

print('=== FIRST AND LAST PARAMETER VALUES ===')
first_key = state_dict_keys[0]
last_key = state_dict_keys[-1]
first_param = checkpoint['model_state_dict'][first_key]
last_param = checkpoint['model_state_dict'][last_key]

print(f'First param ({first_key}):')
print(f'  Shape: {first_param.shape}')
print(f'  Values (first 5): {first_param.flatten()[:5].tolist()}')

print(f'Last param ({last_key}):')
print(f'  Shape: {last_param.shape}')
print(f'  Values (first 5): {last_param.flatten()[:5].tolist()}')
