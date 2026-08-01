# Cloud GPU Setup — Multi-Model Experiments on Vast.ai

Run the full 8-model x 4-dataset experiment matrix on a rented A100-80GB.

---

## Cost & Time Summary

| Plan | GPUs | Time | Cost |
|---|---|---|---|
| **Core experiment** (Qwen family, 3 models x 4 datasets) | 1x A100-80GB | ~4 days | ~$85 |
| **Full matrix** (8 models x 4 datasets) | 1x A100-80GB | ~10 days | ~$225 |
| **Full matrix (parallel)** | 3x A100-80GB | ~3-4 days | ~$290 |

---

## Phase 1: Rent GPU on Vast.ai

### 1. Create account
- Go to https://vast.ai and sign up
- Add credits ($100 to start is safe for the core experiment)

### 2. Search for instance
- Click **Search** in the sidebar
- Filters:
  - **GPU Type**: A100 80GB (or H100 80GB)
  - **GPU RAM**: >= 80 GB
  - **Disk Space**: >= 200 GB
  - **CUDA Version**: >= 12.1
  - **Reliability**: >= 99%
- Sort by **price (low to high)**
- Pick the cheapest one at ~$0.80-1.20/hr

### 3. Launch instance
- **Docker Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- **Disk**: 200 GB
- **Launch mode**: SSH
- Click **Rent**

### 4. Connect
```bash
# Vast.ai gives you an SSH command like:
ssh -p PORT root@IP_ADDRESS -L 8080:localhost:8080
```

---

## Phase 2: Environment Setup (one-time, ~15 min)

Run these commands on the rented GPU:

```bash
# 1. Clone the repo
cd /workspace
git clone https://github.com/kishormorol/agent-repair.git
cd agent-repair

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
# Expected: GPU: NVIDIA A100-SXM4-80GB, VRAM: 80.0 GB

# 5. Verify vLLM
python -c "import vllm; print(f'vLLM {vllm.__version__}')"

# 6. Login to HuggingFace (needed for some gated models like Llama)
pip install huggingface_hub[cli]
huggingface-cli login
# Paste your HF token (get one at https://huggingface.co/settings/tokens)
# You need to accept the Llama 3 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

---

## Phase 3: Smoke Test (~20 min)

Before committing to long runs, verify everything works:

```bash
cd /workspace/agent-repair
source .venv/bin/activate

# Quick test with smallest model, 20 questions
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-7b \
  --dataset hotpotqa \
  --limit 20
```

If it completes and writes to `outputs/hotpotqa/qwen2.5-7b/`, you're good.

---

## Phase 4: Run Experiments

### Recommended order (most informative first)

We run the **Qwen family first** (same family, different sizes) to isolate the
effect of model scale. Then cross-family comparison.

#### Batch 1: Qwen family scaling (core experiment, ~$85)

```bash
cd /workspace/agent-repair
source .venv/bin/activate

# Use tmux so the job survives SSH disconnects
tmux new -s batch1

# Run Qwen 7B on all 4 datasets
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-7b \
  > outputs/logs/qwen7b_all.log 2>&1

# Run Qwen 14B on all 4 datasets
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-14b \
  > outputs/logs/qwen14b_all.log 2>&1

# Run Qwen 72B on all 4 datasets
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-72b \
  > outputs/logs/qwen72b_all.log 2>&1
```

**Detach tmux**: `Ctrl-b d`
**Reattach**: `tmux attach -t batch1`
**Check progress**: `tail -f outputs/logs/qwen7b_all.log`

**Estimated time**: ~96 hours (12 runs x ~8h each)

#### Batch 2: Cross-family comparison (~$60)

```bash
tmux new -s batch2

# Llama family
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model llama-3.1-8b \
  > outputs/logs/llama8b_all.log 2>&1

python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model llama-3.3-70b \
  > outputs/logs/llama70b_all.log 2>&1
```

#### Batch 3: Remaining models (~$80)

```bash
tmux new -s batch3

for model in gemma-2-9b mistral-nemo-12b gemma-2-27b; do
  echo "=== Starting $model ==="
  python scripts/run_experiment.py \
    --config config/config_experiment.yaml \
    --model $model \
    > outputs/logs/${model}_all.log 2>&1
done
```

---

## Phase 5: Monitor Progress

```bash
# Check which runs are done
cat outputs/experiment_log.json | python -m json.tool

# Check GPU utilization
nvidia-smi

# Watch live log
tail -f outputs/logs/qwen7b_all.log

# Check how many trajectories generated
ls outputs/hotpotqa/qwen2.5-7b/trajectories/ | wc -l
```

---

## Phase 6: Download Results

When experiments finish, download results to your MacBook:

```bash
# From your MacBook terminal:
# Option A: scp (replace PORT and IP from Vast.ai)
scp -r -P PORT root@IP_ADDRESS:/workspace/agent-repair/outputs/ \
  ~/agent-repair/outputs/

# Option B: rsync (faster for large dirs, resumable)
rsync -avz --progress -e "ssh -p PORT" \
  root@IP_ADDRESS:/workspace/agent-repair/outputs/ \
  ~/agent-repair/outputs/

# Option C: Push results to git (smaller files only)
# On the GPU machine:
cd /workspace/agent-repair
git add outputs/*/tables/ outputs/*/figures/ results/
git commit -m "Add multi-model experiment results"
git push
```

**Important**: Download results BEFORE destroying the instance!

---

## Phase 7: Cross-Model Analysis (on MacBook)

After downloading results, run the analysis locally (CPU only):

```bash
cd ~/agent-repair
python scripts/run_cross_model_analysis.py  # if this script exists
# OR use the cross-dataset notebook adapted for multi-model
jupyter notebook run_cross_dataset_analysis.ipynb
```

---

## Quick Reference

### Run a single (model, dataset) pair
```bash
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-7b --dataset hotpotqa
```

### Resume after interruption (safe to re-run)
```bash
# Just re-run the same command — checkpoints handle resumption
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-7b
```

### Skip completed stages
```bash
python scripts/run_experiment.py \
  --config config/config_experiment.yaml \
  --model qwen2.5-7b --stages "5,6"  # only repair + eval
```

### See the full matrix without running
```bash
python scripts/run_experiment.py \
  --config config/config_experiment.yaml --dry-run
```

### Reduce batch size if OOM
Edit `config/config_experiment.yaml`:
```yaml
runtime:
  gen_batch_size: 16        # from 32
  repair_batch_size: 16     # from 32
  judge_batch_size: 8       # from 16
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| **OOM on 72B model** | Lower `judge_batch_size` to 4-8 in config |
| **SSH disconnects** | Use `tmux` (see above) |
| **Model download fails** | Run `huggingface-cli login` and accept model licenses |
| **vLLM version mismatch** | `pip install vllm>=0.19.0 --force-reinstall` |
| **Disk full** | Clear HF cache: `rm -rf ~/.cache/huggingface/hub/models--*` after each tier |
| **Vast.ai instance killed** | Re-rent, clone repo, resume (checkpoints are in outputs/) |
| **Want to save money overnight** | Pause the instance in Vast.ai dashboard (keeps disk, stops billing) |

---

## Model License Requirements

Before running, accept these licenses on HuggingFace:

| Model | License Page |
|---|---|
| Llama 3.1-8B | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Llama 3.3-70B | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| Gemma 2 (9B, 27B) | https://huggingface.co/google/gemma-2-9b-it |

Qwen and Mistral models don't require license acceptance.
