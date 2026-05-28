# VeriRAG Technical Documentation

This directory contains the paper-facing technical documentation for the current VeriRAG codebase.

## Documents

- [System Architecture Design](architecture.md)
- [RL Training Pipeline](training_pipeline.md)
- [Data Construction Plan](data_construction.md)
- [Evaluation Protocol](evaluation_protocol.md)
- [Current Mainline](CURRENT_MAINLINE.md)
- [NQ Train / Dev / Test Protocol](NQ_TRAIN_DEV_TEST_PROTOCOL.md)

## Current Research Mainline

The current method should be described as verification-guided evidence control:

```text
Query + Retrieved Docs
  -> Document Risk Scorer
  -> Document-level Evidence Policy
  -> Verification-guided Evidence Controller
  -> Protected Qwen Generation / Abstain
```

The legacy query-level PPO and four-layer verification modules remain available for compatibility and auxiliary signals, but the current paper-facing story is document-level evidence verification before generation.
