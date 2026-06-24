# pi0-action-expert-analysis

> Deconstructing the **π0 action expert** with a representation-analysis toolbox borrowed from systems & computational neuroscience.

**What this is.** A research repo that takes the [openpi](https://github.com/Physical-Intelligence/openpi) π0 vision-language-action model, stands up a reproducible serving + simulation + profiling stack around it, and then probes the *action expert* — the flow-matching module that turns VLM features into robot actions — the way a neuroscientist probes a neural population: latency/throughput characterization, representational geometry, and closed-loop behavior under controlled perturbations.

**Why it's different.** Most VLA work treats the policy as a black box and reports task success. Here the angle is **neuroscience → embodied AI**: apply population-level representation analysis (manifold geometry, dimensionality, decoding) to understand *how* the action expert encodes state and intent, not just *whether* it succeeds.

---

## Status

🚧 Early scaffolding. The skeleton below is in place; sections fill in as experiments land.

| Stage | Component | State |
|---|---|---|
| Serving | policy server + checkpoint cache | scaffolded |
| Smoke tests | GPU inference / preprocessing checks | scaffolded |
| Profiling | latency / throughput / GPU mem | scaffolded |
| Sim | LIBERO closed-loop rollouts | scaffolded |
| Transforms | DROID ↔ LIBERO I/O schema docs | scaffolded |
| Fine-tune | LeRobot conversion + LoRA/full FT | planned |
| Eval harness | multi-task / multi-seed success rate | planned |
| Ablations | pretrained vs FT, LoRA vs full, horizon, norm | planned |

<!-- TODO: drop 1–2 key figures here (latency table, rollout frames, or manifold viz). -->

---

## Repository layout

```
pi0-action-expert-analysis/
├── configs/        # experiment configs (policy, task, seed, horizon)
├── serving/        # policy server launch + checkpoint cache
├── smoke_tests/    # GPU smoke-testing workflow
├── profiling/      # latency / throughput / GPU-mem profiling
├── sim/            # LIBERO/robosuite/MuJoCo closed-loop runner
├── transforms/     # robot I/O transforms (DROID, LIBERO) + tracking doc
├── scripts/        # Slurm submission scripts
├── results/        # latency tables, rollout videos, figures
├── docs/           # experiment log / findings drafts
├── finetune/       # [planned] LeRobot conversion, norm stats, training
├── eval_harness/   # [planned] formal multi-task/seed evaluation
├── ablations/      # [planned] ablation studies
└── third_party/
    └── openpi/     # submodule, pinned at c23745b (read-only)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/jingchengsimon/pi0-action-expert-analysis.git
cd pi0-action-expert-analysis

# if you cloned without --recurse-submodules:
git submodule update --init --recursive
```

openpi is vendored as a submodule pinned at commit `c23745b` so the exact dependency version is auditable.

## License

TBD.
