
import argparse
import json
import random
import numpy as np
import os
import sys
import pickle
import time
import yaml
import copy
from tqdm import tqdm
from model.util import top_auc, cal_hv

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.base import Item, ItemFactory
from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator, generate_initial_population

class ConfigWrapper:
    def __init__(self, data):
        self.data = data
    
    def get(self, key, default=None):
        if not isinstance(key, str):
            return default
            
        keys = key.split('.')
        curr = self.data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr
    
    def __getitem__(self, key):
        return self.get(key)
    
    def to_string(self):
        return str(self.data)

class BaselineOptimizer:
    def __init__(self, config_data, seed, algo_name):
        self.config = ConfigWrapper(config_data)
        self.seed = seed
        self.algo_name = algo_name
        
        # Setup Evaluator
        self.evaluator = SimpleGSCOEvaluator(self.config)
        self.goals = self.config.get('goals', ['f_B', 'f_S', 'I_max'])
        self.item_factory = ItemFactory(self.goals)
        
        self.pop_size = self.config.get('optimization.pop_size', 50)
        self.budget = self.config.get('optimization.eval_budget', 500)
        
        # Grid settings
        self.nPhi = self.config.get('coil_design.wf_nPhi', 12)
        self.nTheta = self.config.get('coil_design.wf_nTheta', 12)
        
        # Constraints
        self.min_cells = self.config.get('llm_constraints.min_active_cells', 3)
        self.max_cells = self.config.get('llm_constraints.max_active_cells', 60)
        
        # Tracking
        self.all_mols = [] # List of (Item, generation_idx)
        self.start_time = time.time()
        self.results_list = [] # For JSON logging
        
        # Seeding
        random.seed(seed)
        np.random.seed(seed)
        
        # Paths
        self.save_dir = os.path.join(self.config.get('save_dir', 'results'), 'baselines')
        os.makedirs(self.save_dir, exist_ok=True)
        self.json_path = os.path.join(self.save_dir, f"{self.algo_name}_{self.seed}_metrics.json")
        self.pkl_path = os.path.join(self.save_dir, f"{self.algo_name}_{self.seed}.pkl")

    def _eval_population(self, items):
        # Evaluate using the provided evaluator
        # Note: evaluate returns (items, log_dict)
        items, _ = self.evaluator.evaluate(items)
        return items

    def _log_metrics(self, current_gen):
        # Filter valid items
        valid_items = [item for item, _ in self.all_mols if item.total is not None]
        if not valid_items:
            return

        # Calculate metrics similar to MOO.py
        total_generated = len(self.all_mols)
        unique_mols = len(set(item.value for item, _ in self.all_mols))
        uniqueness = unique_mols / total_generated if total_generated > 0 else 0.0
        
        # Validity (assuming all are valid structure-wise, but check total)
        valid_count = len(valid_items)
        validity = valid_count / total_generated if total_generated > 0 else 0.0
        
        # Top stats
        sorted_items = sorted(valid_items, key=lambda x: x.total, reverse=True)
        top10 = sorted_items[:10]
        top100 = sorted_items[:100]
        
        avg_top1 = sorted_items[0].total if sorted_items else 0.0
        avg_top10 = np.mean([i.total for i in top10]) if top10 else 0.0
        avg_top100 = np.mean([i.total for i in top100]) if top100 else 0.0
        
        # AUC calculation (using full buffer)
        # Note: top_auc expects list of (item, info) tuples
        auc1 = top_auc(self.all_mols, 1, finish=False, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(self.all_mols, 10, finish=False, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(self.all_mols, 100, finish=False, freq_log=100, max_oracle_calls=self.budget)
        
        # Hypervolume
        if top100:
            scores = np.array([i.scores for i in top100])
            hv = cal_hv(scores)
        else:
            hv = 0.0

        metrics = {
            'all_unique_moles': unique_mols,
            'llm_calls': 0, # Baselines don't use LLM calls
            'Uniqueness': uniqueness,
            'Validity': validity,
            'Training_step': current_gen, # Use gen as step
            'avg_top1': avg_top1,
            'avg_top10': avg_top10,
            'avg_top100': avg_top100,
            'top1_auc': auc1,
            'top10_auc': auc10,
            'top100_auc': auc100,
            'hypervolume': hv,
            'div': 0.0, # Not calculating chemical diversity
            'generated_num': total_generated,
            'running_time[s]': time.time() - self.start_time
        }
        
        self.results_list.append(metrics)
        
        # Save JSON
        with open(self.json_path, 'w') as f:
            json.dump({'params': self.config.to_string(), 'results': self.results_list}, f, indent=4)
            
        print(f"Gen {current_gen}: Best={avg_top1:.4f} | HV={hv:.4f} | Count={total_generated}")

    def _save_results(self):
        data = {
            'all_mols': self.all_mols,
            'config': self.config.data,
            'running_time': time.time() - self.start_time,
            'evaluation': self.results_list
        }
        
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Full state saved to {self.pkl_path}")

    def run(self):
        raise NotImplementedError

class RandomSearch(BaselineOptimizer):
    def run(self):
        print(f"Starting Random Search (Budget: {self.budget})...")
        
        pbar = tqdm(total=self.budget)
        generated = 0
        gen = 0
        
        while generated < self.budget:
            gen += 1
            batch_size = min(self.pop_size, self.budget - generated)
            
            batch_items = []
            for _ in range(batch_size):
                n_active = random.randint(self.min_cells, min(self.max_cells, 20))
                cells = []
                chosen_indices = random.sample(range(self.nPhi * self.nTheta), n_active)
                for idx in chosen_indices:
                    phi = idx // self.nTheta
                    theta = idx % self.nTheta
                    state = random.choice([-1, 1])
                    cells.append([phi, theta, state])
                
                json_str = json.dumps({"cells": cells})
                batch_items.append(self.item_factory.create(json_str))
            
            # Evaluate
            evaluated_items = self._eval_population(batch_items)
            
            # Store
            for item in evaluated_items:
                self.all_mols.append((item, gen))
            
            generated += len(evaluated_items)
            pbar.update(len(evaluated_items))
            
            self._log_metrics(gen)
            self._save_results()
            
        pbar.close()

class StandardGA(BaselineOptimizer):
    def run(self):
        print(f"Starting Standard GA (Budget: {self.budget}, Pop: {self.pop_size})...")
        
        # Initial Population (Warm Start enabled via config)
        init_strs = generate_initial_population(self.config, self.seed)
        # Ensure we have enough
        if len(init_strs) < self.pop_size:
            # fill with random
            needed = self.pop_size - len(init_strs)
            for _ in range(needed):
                n_active = random.randint(self.min_cells, 20)
                cells = []
                chosen_indices = random.sample(range(self.nPhi * self.nTheta), n_active)
                for idx in chosen_indices:
                    phi = idx // self.nTheta
                    theta = idx % self.nTheta
                    state = random.choice([-1, 1])
                    cells.append([phi, theta, state])
                init_strs.append(json.dumps({"cells": cells}))
        
        init_items = [self.item_factory.create(s) for s in init_strs[:self.pop_size]]
        population = self._eval_population(init_items)
        
        # Log initial population
        for item in population:
            self.all_mols.append((item, 0))
            
        gen = 0
        self._log_metrics(gen)
        self._save_results()
        
        eval_count = len(population)
        
        while eval_count < self.budget:
            gen += 1
            # Selection (Tournament)
            parents = self._tournament_selection(population, self.pop_size)
            
            # Crossover & Mutation
            offspring_items = []
            for i in range(0, self.pop_size, 2):
                if i + 1 >= len(parents): break
                p1 = parents[i]
                p2 = parents[i+1]
                
                c1_json, c2_json = self._crossover(p1, p2)
                c1_json = self._mutate(c1_json)
                c2_json = self._mutate(c2_json)
                
                offspring_items.append(self.item_factory.create(c1_json))
                offspring_items.append(self.item_factory.create(c2_json))
            
            # Evaluate Offspring
            # Truncate if exceeding budget
            if eval_count + len(offspring_items) > self.budget:
                offspring_items = offspring_items[:self.budget - eval_count]
            
            if not offspring_items:
                break
                
            evaluated_offspring = self._eval_population(offspring_items)
            eval_count += len(evaluated_offspring)
            
            for item in evaluated_offspring:
                self.all_mols.append((item, gen))
            
            # Replacement (Elitism)
            # Combine pop + offspring, sort by overall_score, take top N
            combined = population + evaluated_offspring
            combined.sort(key=lambda x: x.total, reverse=True) # overall_score is higher better
            population = combined[:self.pop_size]
            
            self._log_metrics(gen)
            self._save_results()

    def _tournament_selection(self, pop, n_parents, k=3):
        parents = []
        for _ in range(n_parents):
            candidates = random.sample(pop, k)
            best = max(candidates, key=lambda x: x.total)
            parents.append(best)
        return parents

    def _crossover(self, p1, p2):
        # One-point crossover on the cell list
        # We need to parse JSON
        c1_data = json.loads(p1.value)['cells']
        c2_data = json.loads(p2.value)['cells']
        
        # Sort cells to make crossover meaningful (e.g. spatial clustering)
        # Sort by phi, then theta
        c1_data.sort(key=lambda x: (x[0], x[1]))
        c2_data.sort(key=lambda x: (x[0], x[1]))
        
        # Cut point
        min_len = min(len(c1_data), len(c2_data))
        if min_len < 2:
            return p1.value, p2.value
            
        cut = random.randint(1, min_len - 1)
        
        new_c1 = c1_data[:cut] + c2_data[cut:]
        new_c2 = c2_data[:cut] + c1_data[cut:]
        
        return json.dumps({"cells": new_c1}), json.dumps({"cells": new_c2})

    def _mutate(self, json_str):
        # Same mutation logic as GA, reusing code would be better but copy-paste for standalone
        # I'll instantiate StandardGA locally to use its mutate or just copy
        # Copying minimal logic
        data = json.loads(json_str)
        cells = data['cells']
        mut_type = random.choice([1, 2, 3, 4])
        
        if mut_type == 1 and cells: # Flip
            idx = random.randrange(len(cells))
            cells[idx][2] *= -1 # Flip sign
            
        elif mut_type == 2 and cells: # Move
            idx = random.randrange(len(cells))
            # Random walk
            cells[idx][0] = (cells[idx][0] + random.choice([-1, 0, 1])) % self.nPhi
            cells[idx][1] = (cells[idx][1] + random.choice([-1, 0, 1])) % self.nTheta
            
        elif mut_type == 3: # Add
            if len(cells) < self.max_cells:
                phi = random.randint(0, self.nPhi - 1)
                theta = random.randint(0, self.nTheta - 1)
                state = random.choice([-1, 1])
                cells.append([phi, theta, state])
                
        elif mut_type == 4: # Remove
            if len(cells) > self.min_cells:
                idx = random.randrange(len(cells))
                cells.pop(idx)
        
        # Deduplicate: if multiple cells at same pos, sum them or keep last.
        # GSCO-Lite logic says only -1,0,1. 
        # Let's simplify: use a dict map
        cell_map = {}
        for c in cells:
            k = (c[0], c[1])
            cell_map[k] = c[2] # Last write wins
        
        final_cells = [[k[0], k[1], v] for k, v in cell_map.items() if v != 0]
        
        return json.dumps({"cells": final_cells})

class SimulatedAnnealing(BaselineOptimizer):
    def run(self):
        print(f"Starting Simulated Annealing (Budget: {self.budget})...")
        
        # Initial Solution
        init_strs = generate_initial_population(self.config, self.seed)
        start_str = init_strs[0] if init_strs else json.dumps({"cells": []})
        
        curr_item = self.item_factory.create(start_str)
        curr_item = self._eval_population([curr_item])[0]
        
        self.all_mols.append((curr_item, 0))
        
        best_item = curr_item
        
        eval_count = 1
        T = 1.0
        T_min = 0.01
        alpha = 0.95
        gen = 0
        
        # Log initial
        self._log_metrics(gen)
        self._save_results()
        
        while eval_count < self.budget:
            gen += 1
            # Mutate
            neighbor_json = self._mutate(curr_item.value)
            neighbor_item = self.item_factory.create(neighbor_json)
            neighbor_item = self._eval_population([neighbor_item])[0]
            eval_count += 1
            self.all_mols.append((neighbor_item, gen))
            
            # Accept?
            delta = neighbor_item.total - curr_item.total
            
            if delta > 0:
                curr_item = neighbor_item
                if curr_item.total > best_item.total:
                    best_item = curr_item
            else:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    curr_item = neighbor_item
            
            # Cool down
            T = max(T_min, T * alpha)
            
            if eval_count % 10 == 0:
                 self._log_metrics(gen)
                 self._save_results()

    def _mutate(self, json_str):
        # Reusing mutation logic (copied for standalone)
        data = json.loads(json_str)
        cells = data['cells']
        mut_type = random.choice([1, 2, 3, 4])
        
        if mut_type == 1 and cells:
            idx = random.randrange(len(cells))
            cells[idx][2] *= -1 
        elif mut_type == 2 and cells: 
            idx = random.randrange(len(cells))
            cells[idx][0] = (cells[idx][0] + random.choice([-1, 0, 1])) % self.nPhi
            cells[idx][1] = (cells[idx][1] + random.choice([-1, 0, 1])) % self.nTheta
        elif mut_type == 3: 
            if len(cells) < self.max_cells:
                cells.append([random.randint(0, self.nPhi - 1), random.randint(0, self.nTheta - 1), random.choice([-1, 1])])
        elif mut_type == 4: 
            if len(cells) > self.min_cells:
                cells.pop(random.randrange(len(cells)))
        
        cell_map = {}
        for c in cells: cell_map[(c[0], c[1])] = c[2]
        final_cells = [[k[0], k[1], v] for k, v in cell_map.items() if v != 0]
        return json.dumps({"cells": final_cells})

class GreedySearch(BaselineOptimizer):
    def __init__(self, config_data, seed, lambda_S=1e-6):
        super().__init__(config_data, seed, 'GreedyGSCO')
        self.lambda_S = lambda_S
        
    def run(self):
        print(f"Starting Greedy GSCO Search (Budget: {self.budget}, Lambda_S: {self.lambda_S})...")
        
        # Initialize Gradient Helper in Evaluator
        self.evaluator._init_gradient_helper()
        if not self.evaluator.gradient_helper_initialized:
            print("Error: Could not initialize gradient helper. Aborting Greedy Search.")
            return

        # Start with an empty configuration
        current_cells = [] 
        cell_map = {}
        
        # Manual initialization for Step 0 (Empty Grid) to bypass evaluator validation
        if self.evaluator.B_ext_n is not None:
            current_B_total = self.evaluator.B_ext_n.copy()
        else:
            current_B_total = np.zeros(self.evaluator.n_grid_points)
            
        dS = self.evaluator.dS_flat
        norm = self.evaluator.norm_factor
        resp_matrix = self.evaluator.response_matrix
        
        current_f_B = 0.5 * np.sum(current_B_total**2 * dS) * norm
        current_f_S = 0
        current_f_GSCO = current_f_B + self.lambda_S * current_f_S
        
        print(f"Step 0: Empty Grid | f_GSCO={current_f_GSCO:.6e} | f_B={current_f_B:.6e}")
        
        # Helper to get cell index
        def get_idx(p, t): return p * self.nTheta + t
        
        step = 0
        while step < self.budget:
            step += 1
            
            # --- GREEDY STEP ---
            # Evaluate all possible single-cell modifications
            
            # Cache constants
            dS = self.evaluator.dS_flat
            norm = self.evaluator.norm_factor
            resp_matrix = self.evaluator.response_matrix
            
            current_f_B = 0.5 * np.sum(current_B_total**2 * dS) * norm
            current_f_S = len(cell_map)
            current_f_GSCO = current_f_B + self.lambda_S * current_f_S
            
            best_move = None
            best_f_GSCO = current_f_GSCO
            best_B_total = None
            
            # Iterate all cells
            for phi in range(self.nPhi):
                for theta in range(self.nTheta):
                    idx = get_idx(phi, theta)
                    current_state = cell_map.get((phi, theta), 0)
                    
                    # Test possible new states
                    for new_state in [-1, 0, 1]:
                        if new_state == current_state:
                            continue
                        
                        # Check forbidden
                        if (phi, theta) in self.evaluator.forbidden_cells and new_state != 0:
                            continue
                            
                        # Calculate Delta B
                        delta_state = new_state - current_state
                        delta_B = delta_state * resp_matrix[idx]
                        
                        # New B total
                        new_B_total = current_B_total + delta_B
                        
                        # New f_B
                        new_f_B = 0.5 * np.sum(new_B_total**2 * dS) * norm
                        
                        # New f_S
                        if current_state == 0 and new_state != 0:
                            new_f_S = current_f_S + 1
                        elif current_state != 0 and new_state == 0:
                            new_f_S = current_f_S - 1
                        else:
                            new_f_S = current_f_S # Flip doesn't change sparsity count
                            
                        # New f_GSCO
                        new_f_GSCO = new_f_B + self.lambda_S * new_f_S
                        
                        if new_f_GSCO < best_f_GSCO:
                            best_f_GSCO = new_f_GSCO
                            best_move = (phi, theta, new_state)
                            best_B_total = new_B_total
            
            # Apply best move
            if best_move:
                phi, theta, new_state = best_move
                print(f"  Step {step}: f_GSCO {current_f_GSCO:.6e} -> {best_f_GSCO:.6e} | Mod cell ({phi},{theta}) {cell_map.get((phi,theta),0)}->{new_state}")
                
                # Update State
                if new_state == 0:
                    if (phi, theta) in cell_map:
                        del cell_map[(phi, theta)]
                else:
                    cell_map[(phi, theta)] = new_state
                
                current_B_total = best_B_total
                
                # Log full item every few steps or if improved
                # Convert map to cells list
                cells_list = [[k[0], k[1], v] for k, v in cell_map.items()]
                json_str = json.dumps({"cells": cells_list})
                new_item = self.item_factory.create(json_str)
                
                # Evaluate officially (to get all metrics and proper format)
                new_item = self._eval_population([new_item])[0]
                self.all_mols.append((new_item, step))
                self._log_metrics(step)
                self._save_results()
                
            else:
                print(f"  Step {step}: No improvement found. Converged.")
                break
                
        print("Greedy Search Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['random', 'ga', 'sa', 'greedy'])
    parser.add_argument('--config', type=str, default='problem/stellarator_coil_gsco_lite/config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_S', type=float, default=1e-5, help='Sparsity penalty weight (default: 1e-5)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        
    if args.algo == 'random':
        optimizer = RandomSearch(config_data, args.seed, 'RandomSearch')
    elif args.algo == 'ga':
        optimizer = StandardGA(config_data, args.seed, 'StandardGA')
    elif args.algo == 'sa':
        optimizer = SimulatedAnnealing(config_data, args.seed, 'SimulatedAnnealing')
    elif args.algo == 'greedy':
        optimizer = GreedySearch(config_data, args.seed, lambda_S=args.lambda_S)
        
    optimizer.run()
