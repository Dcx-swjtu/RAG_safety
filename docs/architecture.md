# System Architecture Design

## Motivation

RAG poisoning attacks mainly operate through retrieved evidence, not through the query alone. Once poisoned evidence enters the generator context, a strong LLM may follow the contaminated context even when it has enough internal knowledge to answer correctly. VeriRAG therefore treats defense as evidence exposure control: identify risky documents, preserve useful support evidence, and verify the final context before generation.

The current architecture is not a query-level reject-only verifier. It is a verification-guided document-level defense pipeline.

## High-Level Pipeline

```text
User Query
  + Retrieved Documents
        |
        v
Document Risk Scorer
        |
        v
Document-level Evidence Policy
        |
        v
Verification-guided Evidence Controller
        |
        v
Protected Qwen Generation / Abstain
```

## Main Components

### 1. Document Risk Scorer

Files:

- `verirag/adversarial_doc_scorer.py`
- `verirag/learned_doc_scorer.py`
- `verirag/text_features.py`

The scorer estimates adversarial risk for each retrieved document. It uses query-document and document-document evidence features such as:

- semantic/text similarity
- document rank and source metadata
- style and attack-pattern features
- outlier/cluster signals
- support score
- conflict score

The learned scorer is trained as a document-level adversarial evidence detector. In current experiments it is the strongest simple filtering baseline.

### 2. Document-level Evidence Policy

Files:

- `verirag/nq_doc_features.py`
- `verirag/nq_doc_policy.py`
- `verirag/nq_document_mask_environment.py`
- `verirag/nq_doc_ppo_trainer.py`

The policy operates at the document level. It outputs a keep/drop decision for each retrieved document and an optional global abstain decision.

This design replaces the older query-level action space:

```text
SKIP / LIGHT / DEEP / EXPAND / REJECT
```

with the current evidence-level control interface:

```text
doc_i -> keep/drop
global -> abstain
```

This is better aligned with the threat model because poisoning occurs in individual retrieved documents.

### 3. Verification-guided Evidence Controller

File:

- `verirag/conflict_aware_generation.py`

The controller is the final gate before generation. It receives the original retrieved documents and scorer/policy annotations, then decides which evidence Qwen is allowed to see.

Current verification signals include:

- `attack_prob`: estimated adversarial probability
- `support_score`: whether the document supports answering the query
- `conflict_score`: whether the document conflicts with other evidence or expected support
- `doc_scorer_kept`: scorer-level filtering decision
- `nq_doc_policy_kept`: policy-level filtering decision
- `nq_doc_policy_keep_prob`: policy confidence

The controller has two important behaviors:

1. Hard-drop high-risk evidence.
2. Rescue high-support clean evidence when risk/conflict are below rescue thresholds.

This fixes the previous failure mode where the policy achieved low ASR but damaged clean accuracy by dropping useful clean evidence.

### 4. Protected Generation

File:

- `verirag/generator.py`

The generator should not be treated as the defense itself. It receives a protected evidence context after the scorer, policy, and verification-guided controller have reduced exposure to poisoned documents.

### 5. Pipeline Orchestration

File:

- `verirag/defense_orchestrator.py`

The orchestrator keeps compatibility with older claim extraction and cross-validation modules, but the current main path is:

```text
retrieve -> score docs -> policy mask -> verify evidence -> generate
```

## Active vs Legacy Boundary

Active paper-facing modules:

- `adversarial_doc_scorer.py`
- `learned_doc_scorer.py`
- `nq_doc_features.py`
- `nq_doc_policy.py`
- `nq_document_mask_environment.py`
- `nq_doc_ppo_trainer.py`
- `conflict_aware_generation.py`
- `defense_orchestrator.py`
- `generator.py`

Legacy or auxiliary modules:

- `policy_network.py`
- `ppo_trainer.py`
- `state_encoder.py`
- `reward_function.py`
- `attack_simulator.py`
- `environment.py`
- `fixed_attack_environment.py`

The legacy modules are still importable and tested, but should not be presented as the primary current method.

## Current Main Result

On the official-mixed NQ500 benchmark with unified Qwen generation:

| Method | ACC | ASR | F1 | CleanDrop |
|---|---:|---:|---:|---:|
| Vanilla RAG | 0.5060 | 0.0940 | 0.6493 | 0.0000 |
| Learned Scorer | 0.5040 | 0.0340 | 0.6624 | 0.0172 |
| SeCon-RAG | 0.5040 | 0.0380 | 0.6615 | 0.0136 |
| Ours | 0.5080 | 0.0144 | 0.6704 | 0.0140 |

The main architectural gain is low ASR without increasing clean evidence damage.
