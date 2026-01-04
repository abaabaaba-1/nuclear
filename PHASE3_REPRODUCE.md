# Phase 3 Reproducibility (VMEC + GSCO)

This document lists the exact commands used to reproduce the Phase 3 results and regenerate the figure/json artifacts under `analysis_outputs/figure/`.

## VMEC (v11 stress) — seeds 42/43/44

### Runs

HeuRepair ON:

```bash
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_on.yaml --seed 42 --run_id Phase3_Stress_VMEC_v11_HeuOn_seed42
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_on.yaml --seed 43 --run_id Phase3_Stress_VMEC_v11_HeuOn_seed43
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_on.yaml --seed 44 --run_id Phase3_Stress_VMEC_v11_HeuOn_seed44
```

HeuRepair OFF:

```bash
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_off.yaml --seed 42 --run_id Phase3_Stress_VMEC_v11_HeuOff_seed42
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_off.yaml --seed 43 --run_id Phase3_Stress_VMEC_v11_HeuOff_seed43
python run_fusionopt.py problem/stellarator_vmec/config_fusionopt_v1_stress_v11_heu_repair_off.yaml --seed 44 --run_id Phase3_Stress_VMEC_v11_HeuOff_seed44
```

### Protocol checks

```bash
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed42
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed43
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed44
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed42
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed43
python check_phase0_protocol.py --results_dir results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed44
```

### Per-seed HeuOn vs HeuOff figures

```bash
python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed42 \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed42 \
  --label-a heu_on --label-b heu_off --max-eval 40 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed42

python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed43 \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed43 \
  --label-a heu_on --label-b heu_off --max-eval 40 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed43

python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_on/Phase3_Stress_VMEC_v11_HeuOn_seed44 \
  results/stellarator_vmec/fusionopt_v1_stress_v11_heu_off/Phase3_Stress_VMEC_v11_HeuOff_seed44 \
  --label-a heu_on --label-b heu_off --max-eval 40 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed44
```

### VMEC multiseed summary

```bash
python analysis/phase3_multiseed_summary.py \
  --inputs \
    analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed42.json \
    analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed43.json \
    analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed44.json \
  --out-prefix analysis_outputs/figure/phase3_stress_vmec_v11_multiseed_summary \
  --label-a heu_on --label-b heu_off
```

## GSCO-Lite (stress v3) — seeds 42/43/44

Stress v3 config files:

- `problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_on.yaml`
- `problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_off.yaml`

### Runs

HeuRepair ON:

```bash
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_on.yaml --seed 42 --run_id Phase3_Stress_GSCO_v3_HeuOn_seed42
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_on.yaml --seed 43 --run_id Phase3_Stress_GSCO_v3_HeuOn_seed43
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_on.yaml --seed 44 --run_id Phase3_Stress_GSCO_v3_HeuOn_seed44
```

HeuRepair OFF:

```bash
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_off.yaml --seed 42 --run_id Phase3_Stress_GSCO_v3_HeuOff_seed42
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_off.yaml --seed 43 --run_id Phase3_Stress_GSCO_v3_HeuOff_seed43
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_fusionopt_v1_stress_v3_heu_repair_off.yaml --seed 44 --run_id Phase3_Stress_GSCO_v3_HeuOff_seed44
```

### Protocol checks

```bash
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed42
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed43
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed44
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed42
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed43
python check_phase0_protocol.py --results_dir results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed44
```

### Per-seed HeuOn vs HeuOff figures

```bash
python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed42 \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed42 \
  --label-a heu_on --label-b heu_off --max-eval 80 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed42

python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed43 \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed43 \
  --label-a heu_on --label-b heu_off --max-eval 80 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed43

python analysis/phase3_heu_repair_ablation.py \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_on/Phase3_Stress_GSCO_v3_HeuOn_seed44 \
  results/stellarator_coil_gsco_lite/fusionopt_v1_stress_v3_heu_off/Phase3_Stress_GSCO_v3_HeuOff_seed44 \
  --label-a heu_on --label-b heu_off --max-eval 80 --step 10 \
  --out-prefix analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed44
```

### GSCO multiseed summary

```bash
python analysis/phase3_multiseed_summary.py \
  --inputs \
    analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed42.json \
    analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed43.json \
    analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed44.json \
  --out-prefix analysis_outputs/figure/phase3_stress_gsco_v3_multiseed_summary \
  --label-a heu_on --label-b heu_off
```

## Artifacts (expected)

VMEC v11:

- `analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed42.{png,json}`
- `analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed43.{png,json}`
- `analysis_outputs/figure/phase3_stress_vmec_v11_heu_on_vs_off_seed44.{png,json}`
- `analysis_outputs/figure/phase3_stress_vmec_v11_multiseed_summary.{png,json}`

GSCO v3:

- `analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed42.{png,json}`
- `analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed43.{png,json}`
- `analysis_outputs/figure/phase3_stress_gsco_v3_heu_on_vs_off_seed44.{png,json}`
- `analysis_outputs/figure/phase3_stress_gsco_v3_multiseed_summary.{png,json}`
