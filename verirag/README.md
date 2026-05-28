# Package Map

## Active Mainline

- `adversarial_doc_scorer.py`
- `learned_doc_scorer.py`
- `text_features.py`
- `nq_doc_features.py`
- `nq_doc_policy.py`
- `nq_document_mask_environment.py`
- `nq_doc_ppo_trainer.py`
- `conflict_aware_generation.py`
- `defense_orchestrator.py`
- `generator.py`

## Compatibility / Legacy

- `claim_extractor.py`
- `cross_validator.py`
- `policy_network.py`
- `state_encoder.py`
- `reward_function.py`
- `environment.py`
- `fixed_attack_environment.py`
- `ppo_trainer.py`
- `attack_simulator.py`

These compatibility modules remain importable because existing tests, older
checkpoints, and historical scripts still use them. New paper-facing work should
prefer the active mainline files above.
