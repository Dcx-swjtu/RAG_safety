"""
Retriever implementations for VeriRAG.

DenseRetriever uses sentence-transformers with optional FAISS acceleration when
available. If those heavy pieces are not available locally, it automatically
falls back to a deterministic lexical retriever. This keeps the engineering
pipeline usable on CPU-only and offline machines.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"\b\w+\b", text) if len(token) > 1]


@dataclass
class RetrievedDocument:
    """Single retrieval result."""

    doc_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DenseRetriever:
    """
    Hybrid dense/lexical retriever with a stable API.

    Args:
        model_path: Local SentenceTransformer path or model id.
        backend: "auto", "dense", or "lexical".
        normalize_embeddings: Whether to use cosine-normalized vectors.
    """

    def __init__(
        self,
        model_path: str = "./models/contriever",
        device: str = "cuda",
        backend: str = "auto",
        normalize_embeddings: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.backend = "lexical"
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []
        self._idf: Dict[str, float] = {}

        if backend in {"auto", "dense"}:
            try:
                from sentence_transformers import SentenceTransformer

                local_candidate = os.path.exists(model_path)
                if backend == "dense" or local_candidate:
                    self.model = SentenceTransformer(model_path, device=device)
                    self.backend = "dense"
            except Exception as exc:
                if backend == "dense":
                    raise RuntimeError(f"Failed to initialize dense retriever: {exc}") from exc

    def build_index(
        self,
        documents: Sequence[Any],
        doc_ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: int = 128,
    ) -> None:
        """Build retrieval index from strings or document dictionaries."""
        texts, ids, metas = self._normalize_documents(documents, doc_ids, metadatas)
        self.documents = texts
        self.doc_ids = ids
        self.metadatas = metas

        if self.backend == "dense" and self.model is not None:
            embeddings = self.model.encode(
                list(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size,
            ).astype("float32")
            if self.normalize_embeddings:
                embeddings = self._normalize(embeddings)
            self.embeddings = embeddings
            self._build_faiss_if_available(embeddings)
        else:
            self._build_lexical_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_documents: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Retrieve top-k document indices and scores.

        Set return_documents=True to receive a list of RetrievedDocument objects.
        """
        if not self.documents:
            return [] if return_documents else ([], [])

        top_k = max(1, min(top_k, len(self.documents)))
        if self.backend == "dense" and self.embeddings is not None and self.model is not None:
            indices, scores = self._dense_search(query, top_k)
        else:
            indices, scores = self._lexical_search(query, top_k)

        if return_documents:
            return self._format_results(indices, scores)  # type: ignore[return-value]
        return indices, scores

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """Return structured retrieval results."""
        return self.retrieve(query, top_k=top_k, return_documents=True)  # type: ignore[return-value]

    def save_index(self, path: str) -> None:
        """Persist the index metadata and embeddings when present."""
        os.makedirs(path, exist_ok=True)
        payload = {
            "backend": self.backend,
            "model_path": self.model_path,
            "normalize_embeddings": self.normalize_embeddings,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "metadatas": self.metadatas,
        }
        with open(os.path.join(path, "retriever.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self.embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        if self.index is not None:
            try:
                import faiss

                faiss.write_index(self.index, os.path.join(path, "faiss.index"))
            except Exception:
                pass

    def load_index(self, path: str) -> None:
        """Load a previously saved index."""
        with open(os.path.join(path, "retriever.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.backend = payload.get("backend", "lexical")
        self.model_path = payload.get("model_path", self.model_path)
        self.normalize_embeddings = payload.get("normalize_embeddings", True)
        self.documents = payload.get("documents", [])
        self.doc_ids = payload.get("doc_ids", [str(i) for i in range(len(self.documents))])
        self.metadatas = payload.get("metadatas", [{} for _ in self.documents])

        embeddings_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path).astype("float32")
            self._build_faiss_if_available(self.embeddings)
        else:
            self._build_lexical_index()

    def _dense_search(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        query_embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype("float32")
        if self.normalize_embeddings:
            query_embedding = self._normalize(query_embedding)

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            return indices[0].tolist(), scores[0].tolist()

        scores = np.dot(self.embeddings, query_embedding[0])
        indices = np.argsort(-scores)[:top_k]
        return indices.tolist(), scores[indices].astype(float).tolist()

    def _lexical_search(self, query: str, top_k: int) -> Tuple[List[int], List[float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return list(range(min(top_k, len(self.documents)))), [0.0] * min(top_k, len(self.documents))

        scores = []
        query_tf = self._term_frequency(query_tokens)
        for doc_tokens in self._doc_tokens:
            doc_tf = self._term_frequency(doc_tokens)
            score = 0.0
            for token, q_count in query_tf.items():
                score += q_count * doc_tf.get(token, 0.0) * self._idf.get(token, 0.0)
            # Small overlap bonus improves behavior for short factoid queries.
            overlap = len(set(query_tokens) & set(doc_tokens))
            score += overlap / max(len(set(query_tokens)), 1)
            scores.append(score)

        scores_array = np.array(scores, dtype="float32")
        indices = np.argsort(-scores_array)[:top_k]
        return indices.tolist(), scores_array[indices].astype(float).tolist()

    def _build_lexical_index(self) -> None:
        self.backend = "lexical" if self.model is None else self.backend
        self._doc_tokens = [_tokenize(doc) for doc in self.documents]
        document_frequency: Dict[str, int] = {}
        for tokens in self._doc_tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        n_docs = max(len(self._doc_tokens), 1)
        self._idf = {
            token: math.log((n_docs + 1) / (df + 1)) + 1.0
            for token, df in document_frequency.items()
        }

    def _build_faiss_if_available(self, embeddings: np.ndarray) -> None:
        try:
            import faiss

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        except Exception:
            self.index = None

    def _format_results(self, indices: List[int], scores: List[float]) -> List[RetrievedDocument]:
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(
                RetrievedDocument(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=float(score),
                    rank=rank,
                    metadata=self.metadatas[idx],
                )
            )
        return results

    @staticmethod
    def _normalize_documents(
        documents: Sequence[Any],
        doc_ids: Optional[Sequence[str]],
        metadatas: Optional[Sequence[Dict[str, Any]]],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        texts: List[str] = []
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = str(doc.get("text", doc.get("content", doc.get("document", ""))))
                doc_id = str(doc.get("doc_id", doc.get("id", idx)))
                metadata = dict(doc.get("metadata", {}))
                if "source" in doc:
                    metadata["source"] = doc["source"]
            else:
                text = str(doc)
                doc_id = str(doc_ids[idx]) if doc_ids is not None else str(idx)
                metadata = dict(metadatas[idx]) if metadatas is not None else {}
            texts.append(text)
            ids.append(doc_id)
            metas.append(metadata)
        return texts, ids, metas

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    @staticmethod
    def _term_frequency(tokens: Iterable[str]) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        total = 0
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
            total += 1
        if total == 0:
            return counts
        return {token: count / total for token, count in counts.items()}


LexicalRetriever = DenseRetriever
