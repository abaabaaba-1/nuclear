#!/usr/bin/env python3
import sys, json
sys.path.insert(0, '/home/dataset-assist-0/MOLLM-main')
from model.MOLLM import ConfigLoader
from problem.stellarator_coil_gsco_lite.evaluator import SimpleGSCOEvaluator

config = ConfigLoader('stellarator_coil_gsco_lite/config.yaml')
evaluator = SimpleGSCOEvaluator(config)

class TestItem:
    def __init__(self, value):
        self.value = value
        self.property = {}
    def assign_results(self, results):
        self.property = results['original_results']

# 测试各种配置
configs = {
    'No coil': [],
    '2 cells': [[3, 5, 1], [8, 2, -1]],
    '5 cells': [[i*2, i*2, 1] for i in range(5)],
    '10 cells': [[i, i, 1 if i%2==0 else -1] for i in range(10)],
    '20 cells': [[i, i, 1 if i%2==0 else -1] for i in range(20)]
}

print('='*70)
print('f_B comparison:')
print('='*70)
f_B_values = {}
for name, cells in configs.items():
    cfg = {'cells': cells}
    item = TestItem(json.dumps(cfg))
    items, _ = evaluator.evaluate([item])
    f_B = items[0].property['f_B'] if items else 0
    f_B_values[name] = f_B
    print(f'{name:15s}: {f_B:.4e} T²m²')

print()
print('Improvements vs baseline:')
baseline = f_B_values['No coil']
for name, f_B in f_B_values.items():
    if name != 'No coil':
        improvement = (baseline - f_B) / baseline * 100
        print(f'{name:15s}: {improvement:+.1f}%')
