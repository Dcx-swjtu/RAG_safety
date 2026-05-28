"""
VeriRAG: document-level RAG defense with conflict-aware generation.

Main path:
- adversarial_doc_scorer / learned_doc_scorer: document risk scoring
- nq_doc_features / nq_doc_policy: per-document keep/drop/abstain policy
- conflict_aware_generation: final evidence control before generation
- defense_orchestrator: end-to-end pipeline orchestration

Claim extraction, cross validation, and query-level PPO remain available as
auxiliary audit and conflict-signal modules.
"""

__version__ = "1.0.0"

from .claim_extractor import ClaimExtractor, Claim, ClaimType
from .cross_validator import CrossValidator, ValidationReport, ConflictDetail
from .policy_network import VerificationPolicyNetwork
from .state_encoder import StateEncoder
from .attack_simulator import AttackSimulator
from .defense_orchestrator import DefenseOrchestrator
from .reward_function import RewardFunction
from .environment import RAGDefenseEnv
from .ppo_trainer import PPOTrainer
from .generator import QwenGenerator, GenerationConfig
from .retriever import DenseRetriever, LexicalRetriever, RetrievedDocument
from .learned_doc_scorer import LearnedAdversarialDocScorer, AdversarialDocClassifier
from .conflict_aware_generation import ConflictAwareEvidenceController, ConflictAwareResult

__all__ = [
    "ClaimExtractor",
    "Claim",
    "ClaimType",
    "CrossValidator",
    "ValidationReport",
    "ConflictDetail",
    "VerificationPolicyNetwork",
    "StateEncoder",
    "AttackSimulator",
    "DefenseOrchestrator",
    "RewardFunction",
    "RAGDefenseEnv",
    "PPOTrainer",
    "QwenGenerator",
    "GenerationConfig",
    "DenseRetriever",
    "LexicalRetriever",
    "RetrievedDocument",
    "LearnedAdversarialDocScorer",
    "AdversarialDocClassifier",
    "ConflictAwareEvidenceController",
    "ConflictAwareResult",
]
