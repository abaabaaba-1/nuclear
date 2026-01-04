# Phase 4 Reproduce: FusionOpt LLM-SemOp (SemOp + Async Batch + Cache + Full Logs)

## Security note (API key)

- Never paste API keys into chat logs.
- Never write API keys into `problem/*.yaml`.
- Use an environment variable instead. The code reads (in order):
  - `FUSIONOPT_API_KEY`
  - `GEMINI_API_KEY`
  - `GOOGLE_API_KEY`
  - `OPENAI_API_KEY`

You can also use `model.api_key_file` to load the key from a local file (recommended for running via tools that may not inherit your shell env). The file is read at runtime and is not logged.

## 1) Smoke run (no API key; SemOp should skip + fallback)

### VMEC

```bash
python run_fusionopt.py problem/stellarator_vmec/config_phase4_llm_semop_smoke.yaml --seed 42
```

### GSCO-Lite

```bash
python run_fusionopt.py problem/stellarator_coil_gsco_lite/config_phase4_llm_semop_smoke.yaml --seed 42
```

The command prints `run_dir:`. Use it for protocol checking.

## 1.5) Low-budget Phase4 comparison (budget=200)

### Baseline (no SemOp)

```bash
python run_fusionopt.py problem/stellarator_vmec/config_phase4_baseline_budget200.yaml --seed 42
```

### LLM-SemOp (real LLM; API key via env var)

In your shell (example):

```bash
export FUSIONOPT_API_KEY='<YOUR_KEY>'
```

Then run:

```bash
python run_fusionopt.py problem/stellarator_vmec/config_phase4_llm_semop_budget200.yaml --seed 42
```

If you prefer a key file instead:

```bash
mkdir -p ~/.config/fusionopt
chmod 700 ~/.config/fusionopt
printf '%s' "<YOUR_KEY>" > ~/.config/fusionopt/api_key
chmod 600 ~/.config/fusionopt/api_key
```

Then keep `model.api_key: ""` in configs and set `model.api_key_file: "~/.config/fusionopt/api_key"`.

Provider note:

- If your key is an OpenAI key (commonly starts with `sk-`), set `model.provider: openai` and `model.name` to an available model (e.g. `gpt-4o-mini`).
- If your key is a Gemini/Google key, keep `model.provider: gemini` and `model.name: gemini-2.5-pro`.

Relay / proxy note (OpenAI-compatible):

- If you are using a company relay that provides an OpenAI-compatible API endpoint, set `model.base_url`.
- Example:

```yaml
model:
  provider: openai
  name: <relay_supported_model_id>
  base_url: "http://35.220.164.252:3888/v1/"
  api_key: ""
  api_key_file: "~/.config/fusionopt/api_key"
```

- Some relays return an empty list for `GET /v1/models`. In that case, you must obtain the correct model id from the relay provider panel/support.

## 2) Protocol check

```bash
python check_phase0_protocol.py --results_dir <run_dir>
```

Expected: `ALL_OK: True`.

## 2.5) Plot curves (HV + feasible rate)

Use the two printed `run_dir` values (baseline vs semop) and run:

```bash
python analysis/phase3_heu_repair_ablation.py <baseline_run_dir> <semop_run_dir> \
  --label-a baseline --label-b llm_semop --max-eval 200 --step 10 \
  --out-prefix analysis_outputs/figure/phase4_llm_semop_budget200_seed42
```

This writes:

- `analysis_outputs/figure/phase4_llm_semop_budget200_seed42.json`
- `analysis_outputs/figure/phase4_llm_semop_budget200_seed42.png`

## 3) Expected output files

Under `results/{problem_id}/{algo_id}/{run_id}/`:

- `evaluations.csv`
- `generations/gen_0000_pop.csv` ...
- `final/pareto_front.csv`
- `run_meta.json`
- `config.yaml` (sanitized: `model.api_key` must be empty)
- `logs/llm_calls.jsonl`
- `cache/llm/` (may be empty if `model.api_key` is empty)

## 4) What to look for in llm_calls.jsonl

With `model.api_key: ""`, you should still see entries like:

- `status: "skipped"`
- `error: "missing_api_key"`
- `operator: "llm_semop"`

This confirms the trigger/batch pipeline executes while the optimizer continues via std operators.
