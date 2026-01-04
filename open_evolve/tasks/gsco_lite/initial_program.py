import random


def propose(rng: random.Random, n: int, nPhi: int, nTheta: int, min_cells: int, max_cells: int, incumbent=None):
    cells_list = []
    for _ in range(max(int(n), 1)):
        if incumbent and rng.random() < 0.6:
            base = list(incumbent)
        else:
            base = []

        cell_map = {(int(c[0]), int(c[1])): int(c[2]) for c in base if isinstance(c, (list, tuple)) and len(c) == 3}

        target_k = rng.randint(int(min_cells), int(max_cells))

        for _ in range(rng.randint(1, 12)):
            if not cell_map or rng.random() < 0.5:
                phi = rng.randrange(int(nPhi))
                theta = rng.randrange(int(nTheta))
                state = rng.choice([-1, 1])
                cell_map[(phi, theta)] = state
            else:
                (phi, theta), state = rng.choice(list(cell_map.items()))
                op = rng.choice(["flip", "move", "remove"])  # noqa: S311
                if op == "flip":
                    cell_map[(phi, theta)] = -1 if state > 0 else 1
                elif op == "move":
                    new_phi = (phi + rng.choice([-1, 0, 1])) % int(nPhi)
                    new_theta = (theta + rng.choice([-1, 0, 1])) % int(nTheta)
                    del cell_map[(phi, theta)]
                    cell_map[(new_phi, new_theta)] = state
                else:
                    del cell_map[(phi, theta)]

        while len(cell_map) < target_k:
            phi = rng.randrange(int(nPhi))
            theta = rng.randrange(int(nTheta))
            if (phi, theta) in cell_map:
                continue
            cell_map[(phi, theta)] = rng.choice([-1, 1])

        while len(cell_map) > target_k:
            k = rng.choice(list(cell_map.keys()))
            del cell_map[k]

        cells = [[p, t, s] for (p, t), s in cell_map.items()]
        cells_list.append(cells)

    return cells_list
