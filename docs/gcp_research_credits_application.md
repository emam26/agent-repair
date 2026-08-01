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

# NAIRR Pilot Start-Up Project Request

**Apply at**: https://submit-nairr.xras.org/opportunities/534233/requests/new
**Decision time**: 2-3 weeks
**Duration**: 3 months
**GPU hours**: Up to 2,000 GPU-hours (select ONE resource)

## Prerequisites

1. Get an **NSF ACCESS account** at https://access-ci.org (same login used for NAIRR portal)
2. Use your **.edu institutional email** (Gmail will be rejected)
3. If you are a graduate student, get a **faculty advisor support letter**

## Recommended Resource Selection

Pick **ONE** (best options for this project):

| Resource | GPU | Max Hours | Best For |
|---|---|---|---|
| **PSC Bridges-2 GPU** | A100-80GB | 2,000 | Large models (70-72B) |
| **SDSC Expanse AI** | A100-40GB | 2,000 | Small/medium models (7-27B) |
| **TACC Vista** | H100 | 480,000 SUs | Fastest inference |

**Recommend: PSC Bridges-2 GPU** — A100-80GB is exactly what we need, and 2,000 hours covers the full experiment matrix (only ~300 needed).

## Project Description (3 pages max, PDF)

Use the content below to create a 3-page PDF. The description must cover: research goals, methodology, compute justification, and expected outcomes.

---

### Page 1: Project Overview and Motivation

**Project Title:** Multi-Model Evaluation of Uncertainty-Guided Error Repair in LLM Agents

**PI:** Md Kishor Morol, [Your Department], [Your University]
**Email:** [your.email]@[university].edu
**Faculty Advisor:** [Advisor Name], [Advisor Email] (if graduate student)

**1. Introduction and Research Goals**

Large language model (LLM) agents that decompose complex tasks into multi-step reasoning trajectories — such as ReAct, Reflexion, and chain-of-thought systems — frequently fail, with even state-of-the-art models producing incorrect answers on 40-85% of multi-hop questions. When failure occurs, practitioners face a critical recovery decision: should the agent restart from scratch, or can it identify and repair only the broken step?

We have developed a systematic framework for studying uncertainty-guided repair strategies in ReAct-style LLM agents, published at AAAI 2025. Our framework evaluates 33 repair strategies using five uncertainty metrics and six localization rules — including three novel cascade-aware rules that account for forward error propagation. Initial experiments with a single 32B model across four reasoning benchmarks revealed three key findings:

1. The optimal repair strategy is task-dependent: full restart outperforms oracle-targeted repair on long trajectories (14.2% vs 10.9% on HotpotQA) due to context contamination, while targeted repair with backtracking wins on shorter ones (3.2% vs 2.8% on FEVER).
2. Backtracking 2 steps upstream consistently improves repair across all tasks, with a dramatic +12.0 percentage point gain on 2WikiMultiHopQA.
3. Uncertainty-guided strategies outperform random baselines by 1.8-6.4 percentage points, validating internal model signals for error localization.

**2. Research Questions for Multi-Model Extension**

Our AAAI 2025 paper identifies a key limitation: all experiments use a single agent model (Qwen2.5-32B). We propose to validate whether these findings generalize across model families and scales by running experiments with 8 open-weight models across 3 capacity tiers:

- RQ1: Does the context contamination effect hold across model scales from 7B to 72B?
- RQ2: Do smaller models benefit more from targeted repair, since they may produce more localized errors?
- RQ3: Are uncertainty-based error localization signals equally informative across model families (Qwen, Llama, Gemma/Mistral)?
- RQ4: How does the repair cost-accuracy tradeoff change with model size?

### Page 2: Methodology and Compute Justification

**3. Methodology**

The experimental pipeline consists of six stages, all fully automated, resumable, and open-source:

| Stage | Function | GPU Required |
|---|---|---|
| 1. Generate | Run ReAct agent on 500 questions per dataset | Yes |
| 2. Uncertainty | Compute 5 step-level uncertainty metrics | Yes |
| 3. Annotate | 72B judge identifies ground-truth broken step | Yes |
| 4. Localize | Apply 30 metric-rule combinations | No (CPU) |
| 5. Repair | Execute 126 repair configurations x 3 seeds | Yes |
| 6. Evaluate | Statistical tests, figures, tables | No (CPU) |

Benchmarks: HotpotQA, FEVER, MuSiQue, 2WikiMultiHopQA (500 stratified questions each, offline document collections for reproducibility).

**4. Compute Justification**

We request **2,000 GPU-hours on PSC Bridges-2 (A100-80GB)**.

Experiment matrix: 8 models x 4 datasets = 32 runs.

| Model Tier | Models | Hours/Run | Total Hours |
|---|---|---|---|
| Small (7-9B) | Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B | ~6h | 72h |
| Medium (12-27B) | Qwen2.5-14B, Mistral-Nemo-12B, Gemma-2-27B | ~8h | 96h |
| Large (70-72B) | Llama-3.3-70B, Qwen2.5-72B | ~10h | 80h |
| Judge (72B, shared) | Qwen2.5-72B-Instruct-AWQ | included | — |
| **Subtotal** | | | **248h** |
| Buffer (failed runs, debugging, reruns) | | | +52h |
| **Total requested** | | | **300h** |

The 72B judge model requires A100-80GB (40 GB at 4-bit quantization). Smaller agent models fit alongside the judge on a single A100-80GB. All models use AWQ 4-bit quantization via vLLM for memory-efficient inference.

300 hours is well within the 2,000-hour limit and provides comfortable margin for debugging and reruns. The pipeline is checkpointed and resumable, so no compute is wasted on interruptions.

**5. Software Stack**

- Framework: Custom Python pipeline (open-source, https://github.com/kishormorol/agent-repair)
- Inference: vLLM >= 0.19.0 with AWQ 4-bit quantization
- Models: Open-weight models from HuggingFace (Qwen, Llama, Gemma, Mistral)
- Dependencies: PyTorch 2.4+, transformers, datasets, scipy, statsmodels
- Container: Compatible with standard CUDA 12.x environments

### Page 3: Expected Outcomes and Broader Impact

**6. Expected Outcomes**

This project will produce:

1. **Empirical results** across 8 models establishing whether the context contamination effect, backtracking benefit, and uncertainty-guided localization generalize across model scales and families.
2. **Scaling analysis** showing how model capacity affects repair strategy effectiveness — informing practitioners on when to invest in targeted repair vs. simply using a larger model.
3. **Cross-architecture comparison** revealing whether some model families produce better-calibrated uncertainty signals for error localization.
4. **Publication**: Results will be included in a follow-up paper extending our AAAI 2025 work, submitted to a top AI/SE venue.
5. **Open-source release**: All trajectories, repair results, and analysis code will be publicly released.

**7. Alignment with NAIRR Focus Areas**

This project directly addresses two NAIRR cross-cutting themes:

- **AI methods for scientific discovery and interpretability**: We study how AI agents reason, fail, and recover — improving interpretability of agent decision-making through uncertainty quantification.
- **Open-source AI tools and datasets**: All code, data, and results are open-source, enabling the broader research community to build on our findings.

**8. Broader Impact**

Understanding error recovery in LLM agents has practical implications:
- **Cost reduction**: Choosing the right repair strategy (targeted vs. restart) based on trajectory length can save 30-50% of inference compute.
- **Reliability**: Uncertainty-guided repair makes agents more dependable for deployment in production systems.
- **Agent design**: Our cross-model findings will inform the design of next-generation self-repairing agents.

**9. Prior Support and Publication Record**

- Published at AAAI 2025 (camera-ready)
- 506,000+ repair evaluations completed with existing compute
- Open-source repository: https://github.com/kishormorol/agent-repair
- All project results will be open and publishable, as required by NAIRR

**10. Project Readiness**

The experimental pipeline is fully implemented, tested, and documented. We can begin using allocated resources within days of the award. The pipeline is:
- Fully automated (single command per model-dataset pair)
- Resumable (checkpointed after every item)
- Efficient (deduplication reduces GPU work by 70-90%)

---

## Faculty Advisor Support Letter Template

Ask your advisor to write a brief letter on department letterhead:

```
Dear NAIRR Review Committee,

I am writing to support Md Kishor Morol's application for NAIRR Pilot
Start-Up computing resources. [He/She/They] is a [PhD/MS] student in
[Department] at [University], working under my supervision on
[uncertainty-guided error repair in LLM agents / your description].

[His/Her/Their] work on this topic has already been published at AAAI 2025,
demonstrating both the quality and maturity of this research. The proposed
multi-model experiments are a natural and important extension that requires
GPU resources beyond what our institution can provide.

I confirm that [he/she/they] has the technical expertise to make productive
use of the requested compute resources and will ensure all results are
openly published.

Sincerely,
[Advisor Name]
[Title]
[Department, University]
[Email]
```

---

## Submission Checklist

- [ ] Register for NSF ACCESS account at https://access-ci.org
- [ ] Log in to NAIRR portal at https://submit-nairr.xras.org
- [ ] Select "Start-Up Project Request"
- [ ] Choose resource: **PSC Bridges-2 GPU** (A100-80GB, up to 2,000 hours)
- [ ] Upload 3-page project description PDF (use content above)
- [ ] Upload faculty advisor support letter (if graduate student)
- [ ] Use .edu email address
- [ ] Submit and wait 2-3 weeks for decision

---

# Google Cloud Research Credits

Apply at: https://cloud.google.com/edu/researchers

Use the Project Summary, Research Questions, Compute Requirements, and Prior Results sections from above. GCP-specific details:

| Service | Specification | Usage |
|---|---|---|
| Compute Engine | A100-80GB VMs (a2-ultragpu-1g) | Model inference |
| Cloud Storage | Standard bucket, ~500 GB | Dataset storage and results |

Request $1,000 to cover ~300 GPU-hours at spot pricing ($1.60/hr) plus buffer.

---

# Fallback: Vast.ai (pay-as-you-go)

If grants take too long, rent immediately:
- A100-80GB at $0.80-1.00/hr
- Core Qwen experiment (12 runs): ~$85
- Full matrix (32 runs): ~$225
- Setup guide: see `CLOUD_GPU_SETUP.md`
