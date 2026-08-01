# Google Cloud Research Credits — Application Draft

Use this to apply at: https://cloud.google.com/edu/researchers

---

## Project Title

Uncertainty-Guided Error Repair in Multi-Step Language Model Agents

## Principal Investigator

Md Kishor Morol
[Your Department], [Your University]
[your.email]@[university].edu

## Project Summary (250 words)

Large language model (LLM) agents that decompose tasks into multi-step reasoning trajectories frequently fail, with even state-of-the-art 70B+ models producing incorrect answers on 40-85% of questions. When failure occurs, a fundamental question arises: should the agent restart from scratch, or can it identify and repair only the broken step?

We have developed a systematic framework for studying uncertainty-guided repair strategies in ReAct-style LLM agents, published at AAAI 2025. Our initial experiments with a single 32B model across four reasoning benchmarks (HotpotQA, FEVER, MuSiQue, 2WikiMultiHopQA) revealed that the optimal repair strategy is task-dependent — full restart wins on long trajectories while targeted repair with backtracking succeeds on shorter ones.

We now need to validate whether these findings generalize across model families and scales. Specifically, we plan to run experiments with 8 open-weight models across 3 capacity tiers (7B to 72B) from 3 model families (Qwen, Llama, Gemma/Mistral). This requires A100-80GB GPUs for inference with the larger models (72B parameters at 4-bit quantization).

The multi-model experiments will produce the first comprehensive study of how model scale and architecture affect error localization and trajectory repair — informing the design of more reliable and cost-effective self-repairing AI agents.

All code is open-source (https://github.com/kishormorol/agent-repair) and results will be published in a follow-up paper.

## Research Questions

1. Does the context contamination effect (full restart outperforming targeted repair on long trajectories) hold across model scales from 7B to 72B?
2. Do smaller models benefit more from targeted repair than larger models, since they have weaker reasoning and may produce more localized errors?
3. Are uncertainty-based error localization signals equally informative across model families (Qwen, Llama, Gemma), or do some architectures produce better-calibrated uncertainty?
4. How does the repair cost-accuracy tradeoff change with model size — does scaling the model reduce the need for sophisticated repair strategies?

## Compute Requirements

### Hardware
- **GPU**: A100-80GB (required for 72B models at AWQ 4-bit quantization)
- **Disk**: 200 GB per instance (model weights + trajectories)
- **RAM**: 64 GB+

### Experiment Matrix
- 8 models x 4 datasets = 32 experiment runs
- Each run: ~8-10 hours on A100-80GB
- Total GPU hours: ~300 hours

### Estimated Cost
- A100-80GB on GCP: ~$1.60/hr (spot pricing)
- Total: ~$480 at spot pricing
- Requesting $1,000 to account for failed runs, reruns, and analysis

### Timeline
- Month 1: Small-tier models (7-9B) across all datasets
- Month 2: Medium-tier models (12-27B) across all datasets
- Month 3: Large-tier models (70-72B) across all datasets
- Month 4: Cross-model analysis, paper writing

## Prior Results and Publication Record

- **Published**: "Uncertainty-Guided Error Repair in Multi-Step Language Model Agents: When to Target, When to Restart" — accepted at AAAI 2025
- **Key results with existing compute**:
  - 506,000+ repair evaluations across 126 configurations x 3 seeds
  - Demonstrated that optimal repair strategy is task-dependent
  - Showed uncertainty-guided repair outperforms random baselines by 1.8-6.4 percentage points
  - Introduced three novel cascade-aware localization rules
- **Open-source code**: https://github.com/kishormorol/agent-repair
- **Impact**: Findings directly inform the design of self-repairing AI agents, a growing area as LLM agents are deployed in production

## GCP Services Required

| Service | Specification | Usage |
|---|---|---|
| Compute Engine | A100-80GB VMs (a2-ultragpu-1g) | Model inference |
| Cloud Storage | Standard bucket, ~500 GB | Dataset storage and results |
| Artifact Registry | Container images | Reproducible environments |

Alternatively, Vertex AI with A100 instances would also work.

## Data Management

- All datasets are publicly available academic benchmarks (HotpotQA, FEVER, MuSiQue, 2WikiMultiHopQA)
- No private or sensitive data is used
- All model weights are open-source (Qwen, Llama, Gemma, Mistral)
- Results and analysis code will be released publicly

## Broader Impact

This research contributes to making LLM agents more reliable and cost-effective. By understanding when targeted repair works and when full restart is necessary, practitioners can:
- Reduce wasted compute by choosing the right recovery strategy
- Build more transparent agents that can diagnose their own failures
- Design better agent architectures informed by empirical evidence across model scales

---

# Alternate: AWS Educate / Azure for Students

If applying to AWS or Azure instead, use the same content above but note:

**AWS**: Apply at https://aws.amazon.com/education/awseducate/
- Request p4d.24xlarge (A100-80GB) or p5.48xlarge (H100) instances
- Same cost estimate applies

**Azure**: Apply at https://azure.microsoft.com/en-us/free/students/
- Request ND A100 v4 series VMs
- $100 free automatically with .edu email (enough for ~60 GPU hours)
- Apply for additional research credits via Microsoft Research program
