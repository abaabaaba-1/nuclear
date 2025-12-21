
import json
from collections import defaultdict

def sanitize(cells):
    valid_cells_map = {}
    for cell in cells:
        if isinstance(cell, list) and len(cell) == 3:
            phi, theta, state = cell
        elif isinstance(cell, dict):
            phi = cell.get('phi', cell.get('phi_idx', -1))
            theta = cell.get('theta', cell.get('theta_idx', -1))
            state = cell.get('state', cell.get('polarity', 0))
        else:
            continue
        
        valid_cells_map[(int(phi), int(theta))] = int(state)
        
    return [[k[0], k[1], v] for k, v in valid_cells_map.items()]

cells = [
    [0, 0, 1],
    [0, 0, 1],
    [1, 1, -1]
]

print(f"Original: {cells}")
cleaned = sanitize(cells)
print(f"Cleaned: {cleaned}")
assert len(cleaned) == 2
assert [0, 0, 1] in cleaned
assert [1, 1, -1] in cleaned
print("Test passed.")
