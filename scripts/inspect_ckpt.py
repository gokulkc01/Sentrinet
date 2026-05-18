import sys
import torch
from pathlib import Path
p = Path(sys.argv[1])
ck = torch.load(p, map_location='cpu')
print('PATH:', p)
print('CONFIG:', ck.get('config'))
print('STEP:', ck.get('step'))
sd = ck.get('policy_state_dict', {})
print('NUM KEYS:', len(sd))
for i,k in enumerate(list(sd.keys())[:200]):
    print(i, k, getattr(sd[k], 'shape', ''))
