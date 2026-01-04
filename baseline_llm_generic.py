import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
import copy
from tqdm import tqdm
import random
import torch
from functools import partial
import os
from algorithm.base import ItemFactory,HistoryBuffer
from openai import AzureOpenAI
from rdkit import Chem
import json
from eval import get_evaluation
import time
from model.util import *
import importlib
import pickle
from model.LLM import LLM
from typing import List, Dict
import yaml
import argparse

# =====================================================================================
#  第一部分：定义新的通用提示生成器 (GenericPromptBuilder)
#  (这部分代码与之前完全相同)
# =====================================================================================

class GenericPromptBuilder:
    """
    A generic, no-domain-knowledge prompt builder for multi-objective optimization.
    """
    def __init__(self, config):
        self.config = config
        self.properties = config.get('goals')
        self.obj_directions = {prop: config.get('optimization_direction')[i] for i, prop in enumerate(self.properties)}
        self.num_offspring = config.get('optimization.num_offspring', default=2)
        self.prop_to_generic_map = {prop: f"objective_{i+1}" for i, prop in enumerate(self.properties)}
        self.experience = ""
        self.pure_experience = ""
        self.experience_prob = config.get('model.experience_prob', 0.0)
        self.exp_times = 0

    def _get_system_prompt(self) -> str:
        return (
            "You are an optimization assistant. Your task is to analyze solutions represented as "
            "JSON objects and propose new, improved versions based on a set of objectives. "
            "You must strictly follow the specified output format."
        )

    def _anonymize_item(self, item: 'Item') -> Dict:
        anonymized_properties = {
            self.prop_to_generic_map[prop]: f"{score:.4f}"
            for prop, score in item.property.items() if prop in self.prop_to_generic_map
        }
        return {
            "solution_definition": item.value,
            "objectives": anonymized_properties
        }

    def _make_objective_statement(self) -> str:
        statements = [
            f"{i+1}. {self.obj_directions[prop].capitalize()} `{self.prop_to_generic_map[prop]}`."
            for i, prop in enumerate(self.properties)
        ]
        return "Your optimization goals are as follows:\n" + "\n".join(statements)

    def _make_instruction_prompt(self) -> str:
        return (
            f"\nBased on the examples provided, generate {self.num_offspring} new and diverse 'solution_definition' JSON strings "
            "that are likely to yield better objective values. Modify the content of the `solution_definition` JSON object.\n\n"
            "CRITICAL: Your output MUST be ONLY the generated JSON content, enclosed within `<candidate>` and `</candidate>` tags. "
            "Do not include any explanations or introductory text.\n\n"
            "Example of a valid output format:\n"
            "<candidate>\n"
            "{\n"
            '    "new_code_blocks": {\n'
            '        "GRUP_LG1": "GRUP LG1         40.000 1.200 ...",\n'
            '        "GRUP_W01": "GRUP W01 W24X94               ...",\n'
            '        "PGRUP_P01": "PGRUP P01 0.5000I..."\n'
            "    }\n"
            "}\n"
            "</candidate>"
        )

    def get_prompt(self, prompt_type: str, ind_list: List['Item'], history_items: List['Item'] = None) -> str:
        full_prompt = [self._get_system_prompt(), "\n---USER---\n"]
        full_prompt.append(self._make_objective_statement())

        use_experience = self.experience and np.random.random() < self.experience_prob
        if use_experience:
            full_prompt.append(f"\nHere is a summary of past findings to guide you:\n<experience>\n{self.pure_experience}\n</experience>\n")

        full_prompt.append("\nHere are some current candidate solutions and their objective values:\n")
        for item in ind_list:
            anonymized_item_str = json.dumps(self._anonymize_item(item), indent=4)
            full_prompt.append(f"```json\n{anonymized_item_str}\n```")

        full_prompt.append(self._make_instruction_prompt())
        return "\n".join(full_prompt)

    def make_experience_prompt(self, all_items: List[tuple]) -> tuple[str, str, str]:
        all_items_clean = [i[0] for i in all_items]
        sorted_items = sorted(all_items_clean, key=lambda x: x.total, reverse=True)
        best_items = sorted_items[:min(10, len(sorted_items))]
        worst_items = sorted_items[-min(10, len(sorted_items)):]
        best_examples_str = "\n".join([json.dumps(self._anonymize_item(item), indent=2) for item in best_items])
        worst_examples_str = "\n".join([json.dumps(self._anonymize_item(item), indent=2) for item in worst_items])
        objective_statement = self._make_objective_statement()

        summary_prompt = (
            f"You are an optimization analyst. Your goal is to summarize key patterns from a set of good and bad solutions.\n\n"
            f"{objective_statement}\n\n"
            f"Analyze the 'solution_definition' in the following 'excellent' solutions:\n{best_examples_str}\n\n"
            f"Analyze the 'solution_definition' in the following 'poor' solutions:\n{worst_examples_str}\n\n"
            "Provide a concise summary (less than 200 words) of what structural patterns in the JSON seem to lead to better results. "
            "Focus on actionable insights about how to modify the `solution_definition`."
        )

        if self.pure_experience:
            summary_prompt += (
                f"\n\nIntegrate these new findings with the previous summary:\n"
                f"<old_experience>\n{self.pure_experience}\n</old_experience>"
            )
        self.exp_times += 1
        return summary_prompt, best_examples_str, worst_examples_str

# =====================================================================================
#  第二部分：克隆的 MOO 类
#  (这部分代码与之前完全相同)
# =====================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MOO:
    # --- MOO 类的完整代码与上一个版本相同 ---
    def __init__(self, reward_system, llm,property_list,config,seed):
        self.reward_system = reward_system
        self.config = config
        self.seed = seed
        self.llm = llm
        self.history = HistoryBuffer()
        self.item_factory = ItemFactory(property_list)
        self.property_list = property_list
        self.pop_size = self.config.get('optimization.pop_size')
        self.budget = self.config.get('optimization.eval_budget')
        self.use_au = self.config.get('use_au')
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
        self.prompt_generator = None
        self.history_moles = []
        self.mol_buffer = []
        self.main_mol_buffer = []
        self.au_mol_buffer = []
        self.results_dict = {'results':[]}
        self.main_results_dict = {'results':[]}
        self.au_results_dict = {'results':[]}
        self.history_experience = []
        self.repeat_num = 0
        self.failed_num = 0
        self.generated_num = 0
        self.llm_calls = 0
        self.patience = 0
        self.old_score = 0
        self.early_stopping = False
        self.record_dict = {}
        for i in ['main','au']:
            for j in ['all_num','failed_num','repeat_num']:
                self.record_dict[i+'_'+j] = 0
        self.record_dict['main_history_smiles'] = []
        self.record_dict['au_history_smiles'] = []
        self.time_step = 0
        self.start_time = time.time()
        self.num_offspring = self.config.get('optimization.num_offspring',default=2)

    def generate_initial_population(self, n):
        module_path = self.config.get('evalutor_path')
        module = importlib.import_module(module_path)
        _generate = getattr(module, "generate_initial_population")
        strings = _generate(self.config,self.seed)
        if isinstance(strings[0],str):
            return [self.item_factory.create(i) for i in strings]
        else:
            self.store_history_moles(strings)
            return strings

    def mutation(self, parent_list):
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles)
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles)
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def evaluate(self,pops):
        pops, log_dict = self.reward_system.evaluate(pops)
        self.failed_num += log_dict.get('invalid_num', 0)
        self.repeat_num += log_dict.get('repeated_num', 0)
        pops = self.store_history_moles(pops)
        return pops

    def store_history_moles(self,pops):
        unique_pop = []
        for i in pops:
            if i.value not in self.history_moles:
                self.history_moles.append(i.value)
                self.mol_buffer.append([i, len(self.mol_buffer)+1])
                unique_pop.append(i)
            else:
                self.repeat_num += 1
        return unique_pop

    def explore(self):
        print("Warning: explore() is not specifically implemented, falling back to mutation-like behavior.")
        if len(self.mol_buffer) > 2:
            parents = random.sample([i[0] for i in self.mol_buffer], 2)
            return self.mutation(parents)
        else:
            return [], "", ""

    def log_results(self, mol_buffer: list = None, buffer_type: str = "default", finish: bool = False) -> None:
        if mol_buffer is None: mol_buffer = self.mol_buffer
        if not mol_buffer:
            print(f"Warning: log_results called with empty buffer for type '{buffer_type}'. Skipping.")
            return
        
        # ... (rest of the log_results method is identical to previous version) ...
        auc1 = top_auc(mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100_mols = [i[0] for i in top100]
        top10 = top100_mols[:10]

        avg_top10 = np.mean([i.total for i in top10]) if top10 else 0.0
        avg_top100 = np.mean([i.total for i in top100_mols]) if top100_mols else 0.0
        avg_top1 = top10[0].total if top10 else 0.0

        if self.config.get('cal_div',default=False):
            from tdc import Evaluator
            div_evaluator = Evaluator(name = 'Diversity')
            diversity_top100 = div_evaluator([i.value for i in top100_mols]) if top100_mols else 0.0
        else:
            diversity_top100 = 0.0
        
        if top100_mols and hasattr(top100_mols[0], 'property') and 'l_delta_b' in top100_mols[0].property and 'aspect_ratio' in top100_mols[0].property:
            all_mols = [item[0] for item in mol_buffer]
            all_mols = [i for i in all_mols if i.constraints['feasibility']<0.01]
            if len(all_mols)>0:
                scores = np.array([[-i.property['l_delta_b'],i.property['aspect_ratio']] for i in all_mols])
                volume = cal_fusion_hv(scores)
            else:
                volume = 0
        else:
            if top100_mols:
                scores = np.array([i.scores for i in top100_mols])
                volume = cal_hv(scores) ###
            else:
                volume = 0.0

        if buffer_type == "default":
            uniqueness = 1 - self.repeat_num / (self.generated_num + 1e-6)
            validity = 1 - self.failed_num / (self.generated_num + 1e-6)
        else:
            uniqueness = 1 - self.record_dict[f'{buffer_type}_repeat_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)
            validity = 1 - self.record_dict[f'{buffer_type}_failed_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)

        if buffer_type == "default":
            new_score = avg_top100
            if new_score - self.old_score < 1e-4 and self.old_score>0.05:
                self.patience += 1
                if self.config.get('early_stopping',default=True) and self.patience >= 6:
                    print('convergence criteria met, abort ...... ')
                    self.early_stopping = True
            else:
                self.patience = 0
            self.old_score = new_score

        if buffer_type == "default": results_dict, save_dir = self.results_dict, os.path.join(self.save_dir, "results")
        elif buffer_type == "main": results_dict, save_dir = self.main_results_dict, os.path.join(self.save_dir, "results_main")
        elif buffer_type == "au": results_dict, save_dir = self.au_results_dict, os.path.join(self.save_dir, "results_au")
        else: raise ValueError(f"Unknown buffer_type: {buffer_type}")

        json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        results_dict['results'].append({
            'all_unique_moles': len(self.history_moles),'llm_calls': self.llm_calls,'Uniqueness': uniqueness,'Validity': validity,
            'Training_step': self.time_step,'avg_top1': avg_top1,'avg_top10': avg_top10,'avg_top100': avg_top100,
            'top1_auc': auc1,'top10_auc': auc10,'top100_auc': auc100,'hypervolume': volume,'div': diversity_top100,
            'input_tokens': self.llm.input_tokens,'output_tokens': self.llm.output_tokens,'generated_num': self.generated_num,
            'running_time[s]': time.time()-self.start_time
        })

        if buffer_type == "default":
            print('================================================================')
            results_dict['params'] = self.config.to_string()
            if len(self.history_experience) > 1: results_dict['history_experience'] = [self.history_experience[0], self.history_experience[-1]]

        with open(json_path, 'w') as f: json.dump(results_dict, f, indent=4)
        
        print(f"{buffer_type}: {len(self.history_moles)}/{self.budget} gen: {self.generated_num} | "
            f"buf: {len(mol_buffer)} | U: {uniqueness:.4f} | V: {validity:.4f} | calls: {self.llm_calls} | "
            f"step: {self.time_step} | top1: {avg_top1:.6f} | top10: {avg_top10:.6f} | top100: {avg_top100:.6f} | "
            f"auc1-10-100: {auc1:.4f}-{auc10:.4f}-{auc100:.4f} | hv: {volume:.4f} | "
            f"unique top100: {len(np.unique([i.value for i in top100_mols])) if top100_mols else 0} | "
            f"tokens(in/out): {self.llm.input_tokens}/{self.llm.output_tokens} | time: {(time.time()-self.start_time)/3600:.3f}h | div: {diversity_top100:.4f}")

    def update_experience(self):
        prompt,_,_ = self.prompt_generator.make_experience_prompt(self.mol_buffer)
        response = self.llm.chat(prompt)
        self.prompt_generator.pure_experience = response
        self.prompt_generator.experience = (f"I have findings from previous attempts: <experience> {response} </experience> Try to use them.\n")
        self.history_experience.append(self.prompt_generator.experience)
        print('length exp:',len(self.prompt_generator.experience))
    
    def run(self):
        store_path = os.path.join(self.save_dir,'mols','_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        if not os.path.exists(os.path.dirname(store_path)): os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        # ... (rest of the run method is identical to previous version) ...
        if self.use_au:
            from genetic_gfn.multi_objective.genetic_gfn.run import Genetic_GFN_Optimizer
            from genetic_gfn.multi_objective.run import prepare_optimization_inputs
            args, config_default, oracle = prepare_optimization_inputs()
            self.au_model = Genetic_GFN_Optimizer(args=args); self.au_model.setup_model(oracle, config_default)
        print('exper_name',self.config.get('exper_name')); set_seed(self.seed); start_time = time.time()
        if self.config.get('inject_per_generation'):
            module = importlib.import_module(self.config.get('evalutor_path')); _get = getattr(module, "get_database")
            database = _get(self.config,n_sample=200)
        if self.config.get('resume'): population,init_pops = self.load_ckpt(store_path)
        else:
            population = self.generate_initial_population(n=self.pop_size)
            if population and population[0].total is None: population = self.evaluate(population)
            if population: self.log_results() # Only log if population is not empty
            init_pops = copy.deepcopy(population)
        
        data = {'history':self.history,'init_pops':init_pops,'final_pops':population,'all_mols':self.mol_buffer,'properties':self.property_list,
                'evaluation': self.results_dict.get('results',[]),'running_time':f'{(time.time()-start_time)/3600:.2f} hours'}
        with open(store_path, 'wb') as f: pickle.dump(data, f)
        
        self.prompt_generator = GenericPromptBuilder(self.config)
        print("✅ [INFO] Using GenericPromptBuilder for 'No-Domain-Knowledge' baseline.")

        self.num_gen = 0
        while True:
            if not population:
                print("Population is empty. Aborting run.")
                break
            if self.config.get('inject_per_generation'): print('inject!'); population.extend(random.sample(database,self.config.get('inject_per_generation')))
            offspring_times = max(min(self.pop_size //self.num_offspring, (self.budget -len(self.mol_buffer)) //self.num_offspring),1)
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(self.pop_size)
            self.log_results()
            if self.config.get('model.experience_prob')>0 and len(self.mol_buffer)>100: self.update_experience()
            if len(self.mol_buffer) >= self.budget or self.early_stopping:
                self.log_results(finish=True)
                if self.use_au: self.log_results(self.main_mol_buffer,buffer_type="main", finish=True); self.log_results(self.au_mol_buffer,buffer_type="au", finish=True)
                break
            self.num_gen+=1
            data = {'history':self.history,'init_pops':init_pops,'final_pops':population,'all_mols':self.mol_buffer,'properties':self.property_list,
                    'evaluation': self.results_dict['results'],'running_time':f'{(time.time()-start_time)/3600:.2f} hours'}
            with open(store_path, 'wb') as f: pickle.dump(data, f)
            if self.num_gen%10==0: print(f"Data saved to {store_path}")
        print(f'=======> total running time { (time.time()-start_time)/3600 :.2f} hours <=======')
        return init_pops,population

    def mating(self, parent_list: list, au: bool = False) -> tuple:
        c, m, e = self.config.get('model.crossover_prob'), self.config.get('model.mutation_prob'), self.config.get('model.explore_prob')
        ops, probs = [], []
        if c > 0: ops.append(self.crossover); probs.append(c)
        if m > 0: ops.append(self.mutation); probs.append(m)
        if e > 0: ops.append(self.explore); probs.append(e)
        
        if not probs: func = self.mutation
        else: func = np.random.choice(ops, p=np.array(probs)/sum(probs))
            
        return func(parent_list)

    def generate_offspring(self, population: list, offspring_times: int) -> list:
        if not population or len(population) < 2:
            print("Warning: Population too small to generate offspring.")
            return []
        
        parents = [random.sample(population, 2) for _ in range(offspring_times)]
        
        # ... (rest of generate_offspring is identical) ...
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.mating, parent_list=p) for p in parents]
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating Offspring"):
                try: results.append(future.result(timeout=900))
                except Exception as e: print(f"Warning: A generation task failed: {e}")
        
        if not results: return []
        children, prompts, responses = zip(*results)
        self.llm_calls += len(results)

        tmp_offspring = [child for pair in children for child in pair]
        self.generated_num += len(tmp_offspring)
        
        # ... (rest of the method)
        if self.use_au:
            self.record(tmp_offspring,'main'); self.save_log_mols(tmp_offspring,buffer_type='main')
            au_items = [self.item_factory.create(s) for s in self.au_model.sample_n_smiles(32,self.mol_buffer)]
            self.generated_num += len(au_items); self.record(au_items,'au'); self.save_log_mols(au_items,buffer_type='au')
            tmp_offspring.extend(au_items)
        
        evaluated_offspring = self.evaluate(tmp_offspring) if tmp_offspring else []
        if prompts and evaluated_offspring: self.history.push(prompts, children, responses)
        return evaluated_offspring
    
    # ... (other methods like save_log_mols, mol_buffer_store, select_next_population, load_ckpt are identical) ...
    def save_log_mols(self, mols: list, buffer_type: str) -> None:
        self.time_step += 1
        mols,_ = self.reward_system.evaluate(mols)
        mol_buffer = self.main_mol_buffer if buffer_type=='main' else self.au_mol_buffer
        self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step,mol_buffer=mol_buffer)
        self.mol_buffer_store(mol_buffer,mols)
        self.log_results(mol_buffer,buffer_type, finish=False)

    def mol_buffer_store(self, mol_buffer: list, mols: list) -> list:
        all_values = {i[0].value for i in mol_buffer}
        for child in mols:
            try:
                mol = Chem.MolFromSmiles(child.value)
                if mol: child.value = Chem.MolToSmiles(mol,canonical=True)
                else: continue
            except: pass
            if child.value not in all_values:
                all_values.add(child.value)
                mol_buffer.append([child,len(self.mol_buffer)+len(mol_buffer)+1])
        return mol_buffer

    def select_next_population(self,pop_size):
        if not self.mol_buffer: return []
        whole_population = [i[0] for i in self.mol_buffer]
        # Fixed: Use proper NSGA-II selection
        return nsga2_selection(whole_population, pop_size) if len(self.property_list)>1 else so_selection(whole_population,pop_size)

    def load_ckpt(self,store_path):
        print('Resuming training...')
        with open(store_path,'rb') as f: ckpt = pickle.load(f)
        json_path = os.path.join(self.save_dir,"results",'_'.join(self.property_list)+'_'+self.config.get('save_suffix')+f'_{self.seed}.json')
        with open(json_path,'r') as f: result_ckpt = json.load(f)
        self.mol_buffer = ckpt.get('all_mols', [])
        population = self.select_next_population(self.pop_size)
        self.history = ckpt.get('history', HistoryBuffer())
        self.history_moles = [i[0].value for i in self.mol_buffer]
        self.results_dict['results'] = ckpt.get('evaluation', [])
        last_result = result_ckpt['results'][-1]
        self.generated_num = last_result.get('generated_num', 0)
        self.llm_calls = last_result.get('llm_calls', 0)
        self.repeat_num = int((1-last_result.get('Uniqueness', 1)) * self.generated_num)
        self.failed_num = int((1-last_result.get('Validity', 1)) * self.generated_num)
        return population, ckpt.get('init_pops', [])

# =====================================================================================
#  第三部分：ConfigLoader 类 (Path-Robust Version)
# =====================================================================================
class ConfigLoader:
    def __init__(self, config_path):
        """
        Initializes the ConfigLoader with a direct path to the config file.
        No path guessing is performed.

        Args:
            config_path (str): The exact path to the configuration YAML file.
        """
        # <<<--- 代码修改开始 ---<<<
        # 删除了之前所有复杂的路径拼接逻辑。
        # 现在它直接使用用户提供的路径。如果路径错误，open()会抛出一个清晰的FileNotFoundError。
        self.config = self._load_config(config_path)
        # <<<--- 代码修改结束 ---<<<

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def to_string(self, config=None, indent=0):
        if config is None: config = self.config
        return yaml.dump(config, indent=indent)


# =====================================================================================
#  第四部分：主执行入口 (if __name__ == "__main__":)
# =====================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run No-Domain-Knowledge LLM Baseline for Multi-Objective Optimization.")
    parser.add_argument('--config', type=str, default='sacs_geo_jk/config.yaml', help='Path to the configuration YAML file (e.g., sacs_geo_jk/config.yaml).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    print("--- Starting Generic LLM Baseline Run ---")
    
    # 1. 加载配置
    print(f"Loading configuration from: {args.config}")
    cfg_path = args.config
    # if cfg_path.startswith('problem/'): cfg_path = cfg_path[len('problem/') :]  <-- Removed
    config = ConfigLoader(cfg_path)

    # 2. 初始化评估系统 (Rewarding System)
    print("Initializing Rewarding System...")
    try:
        module_path = config.get('evalutor_path')
        if not module_path: raise ValueError("'evalutor_path' not found in config.")
        module = importlib.import_module(module_path)
        RewardingSystem = getattr(module, "RewardingSystem")
        reward_system = RewardingSystem(config=config)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"ERROR: Failed to initialize Rewarding System. Check 'evalutor_path' in your config. Details: {e}")
        exit(1)

    # 3. 初始化大语言模型 (LLM)
    print("Initializing Large Language Model...")
    llm = LLM(model=config.get('model.name'), config=config)

    # 4. 获取其他参数
    property_list = config.get('goals')
    seed = args.seed

    # 5. 初始化并运行 MOO
    print("Initializing MOO (Generic Baseline version)...")
    moo_instance = MOO(
        reward_system=reward_system,
        llm=llm,
        property_list=property_list,
        config=config,
        seed=seed
    )

    print(f"\n🚀 Starting MOO run with seed {seed}. Evaluation budget: {config.get('optimization.eval_budget')}. This may take a long time...\n")
    moo_instance.run()

    print("\n--- Generic baseline run completed. ---")
