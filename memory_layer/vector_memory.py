"""
Vector Memory System
Persistent semantic memory for formulas, notes, hypotheses, and research.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import hashlib


class MemoryType(Enum):
    ALPHA_EXPRESSION = "alpha_expression"
    RESEARCH_NOTE = "research_note"
    DISCUSSION = "discussion"
    SIMULATION_RESULT = "simulation_result"
    HYPOTHESIS = "hypothesis"
    FACTOR_DESCRIPTION = "factor_description"
    FAILURE_ANALYSIS = "failure_analysis"
    MACRO_INTERPRETATION = "macro_interpretation"
    CONCEPT = "concept"
    FORMULA_TEMPLATE = "formula_template"


@dataclass
class SemanticMemory:
    """A piece of semantic memory with embedding."""
    id: str
    memory_type: str
    content: str
    embedding_source: str
    created_at: str
    updated_at: str
    metadata: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    related_alpha_id: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0


class VectorMemoryStore:
    """Persistent vector memory store for semantic search."""

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path(__file__).parent / "vector_memory"

        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_file = store_path / "index.json"
        self._load_index()

    def _load_index(self):
        """Load memory index."""
        self.memories: Dict[str, SemanticMemory] = {}
        self.type_index: Dict[str, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}
        self.alpha_index: Dict[str, List[str]] = {}

        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    data = json.load(f)
                    for m_id, m_data in data.get("memories", {}).items():
                        self.memories[m_id] = SemanticMemory(**m_data)

                    self.type_index = data.get("type_index", {})
                    self.tag_index = data.get("tag_index", {})
                    self.alpha_index = data.get("alpha_index", {})
            except Exception:
                pass

    def _save_index(self):
        """Save memory index."""
        data = {
            "memories": {k: asdict(v) for k, v in self.memories.items()},
            "type_index": self.type_index,
            "tag_index": self.tag_index,
            "alpha_index": self.alpha_index
        }

        with open(self.index_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self, content: str, memory_type: str) -> str:
        """Generate unique ID for memory."""
        raw = f"{memory_type}:{content[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Dict = None,
        tags: List[str] = None,
        related_alpha_id: Optional[str] = None,
        importance: float = 0.5
    ) -> str:
        """Add a semantic memory."""
        mem_id = self._generate_id(content, memory_type.value)
        now = datetime.now().isoformat()

        if mem_id in self.memories:
            mem = self.memories[mem_id]
            mem.content = content
            mem.updated_at = now
            mem.access_count += 1
            if metadata:
                mem.metadata.update(metadata)
        else:
            memory = SemanticMemory(
                id=mem_id,
                memory_type=memory_type.value,
                content=content,
                embedding_source=memory_type.value,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
                tags=tags or [],
                related_alpha_id=related_alpha_id,
                importance=importance
            )
            self.memories[mem_id] = memory

            if memory_type.value not in self.type_index:
                self.type_index[memory_type.value] = []
            self.type_index[memory_type.value].append(mem_id)

            for tag in (tags or []):
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(mem_id)

            if related_alpha_id:
                if related_alpha_id not in self.alpha_index:
                    self.alpha_index[related_alpha_id] = []
                self.alpha_index[related_alpha_id].append(mem_id)

        self._save_index()
        return mem_id

    def get_memory(self, mem_id: str) -> Optional[Dict]:
        """Get a specific memory."""
        if mem_id in self.memories:
            mem = self.memories[mem_id]
            mem.access_count += 1
            self._save_index()
            return asdict(mem)
        return None

    def search_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 10
    ) -> List[Dict]:
        """Search memories by type."""
        type_key = memory_type.value
        if type_key not in self.type_index:
            return []

        mem_ids = self.type_index[type_key][:limit]
        return [asdict(self.memories[m_id]) for m_id in mem_ids if m_id in self.memories]

    def search_by_tags(
        self,
        tags: List[str],
        limit: int = 10
    ) -> List[Dict]:
        """Search memories by tags."""
        matching_ids = set()

        for tag in tags:
            if tag in self.tag_index:
                matching_ids.update(self.tag_index[tag])

        results = []
        for m_id in list(matching_ids)[:limit]:
            if m_id in self.memories:
                results.append(asdict(self.memories[m_id]))

        return results

    def search_by_alpha(
        self,
        alpha_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get all memories related to an alpha."""
        if alpha_id not in self.alpha_index:
            return []

        results = []
        for m_id in self.alpha_index[alpha_id][:limit]:
            if m_id in self.memories:
                results.append(asdict(self.memories[m_id]))

        return results

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        return sorted(list(self.tag_index.keys()))

    def get_stats(self) -> Dict:
        """Get memory store statistics."""
        type_counts = {}
        for mem_type, mem_ids in self.type_index.items():
            type_counts[mem_type] = len(mem_ids)

        return {
            "total_memories": len(self.memories),
            "by_type": type_counts,
            "unique_tags": len(self.tag_index),
            "alphas_with_notes": len(self.alpha_index)
        }


class SemanticSearchEngine:
    """Semantic search across vector memories and graph."""

    def __init__(self):
        self.vector_store = VectorMemoryStore()
        self._embed_model = None

    def _get_embed_model(self):
        """Lazy load embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    def semantic_search(
        self,
        query: str,
        k: int = 5,
        memory_types: List[MemoryType] = None,
        tags: List[str] = None
    ) -> List[Dict]:
        """Semantic search across memories."""
        model = self._get_embed_model()
        query_embedding = model.encode([query])[0]

        all_memories = list(self.vector_store.memories.values())

        if memory_types:
            type_values = [mt.value for mt in memory_types]
            all_memories = [m for m in all_memories if m.memory_type in type_values]

        if tags:
            all_memories = [m for m in all_memories if any(t in m.tags for t in tags)]

        if not all_memories:
            return []

        contents = [m.content for m in all_memories]
        embeddings = model.encode(contents)

        scores = []
        for i, mem in enumerate(all_memories):
            from numpy import dot
            from numpy.linalg import norm
            sim = dot(query_embedding, embeddings[i]) / (norm(query_embedding) * norm(embeddings[i]) + 1e-8)
            scores.append((mem, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mem, score in scores[:k]:
            results.append({
                "id": mem.id,
                "memory_type": mem.memory_type,
                "content": mem.content[:200] + "..." if len(mem.content) > 200 else mem.content,
                "score": float(score),
                "tags": mem.tags,
                "related_alpha_id": mem.related_alpha_id
            })

        return results

    def find_similar_alphas(
        self,
        alpha_expression: str,
        alpha_concepts: List[str],
        k: int = 5
    ) -> List[Dict]:
        """Find similar alphas based on expression and concepts."""
        query = f"{alpha_expression} {' '.join(alpha_concepts)}"
        results = self.semantic_search(query, k=k, memory_types=[MemoryType.ALPHA_EXPRESSION])

        similar_alphas = []
        for r in results:
            if r.get("related_alpha_id"):
                similar_alphas.append({
                    "alpha_id": r["related_alpha_id"],
                    "similarity": r["score"],
                    "reason": "similar expression pattern"
                })

        return similar_alphas

    def detect_duplicates(
        self,
        content: str,
        threshold: float = 0.95
    ) -> List[Dict]:
        """Detect potential duplicate experiments."""
        similar = self.semantic_search(content, k=10, memory_types=[
            MemoryType.HYPOTHESIS,
            MemoryType.SIMULATION_RESULT
        ])

        duplicates = []
        for s in similar:
            if s["score"] >= threshold:
                duplicates.append({
                    "memory_id": s["id"],
                    "score": s["score"],
                    "content": s["content"]
                })

        return duplicates

    def cluster_concepts(
        self,
        min_similarity: float = 0.6
    ) -> List[List[str]]:
        """Cluster similar concepts based on memory content."""
        model = self._get_embed_model()

        concept_memories = [
            m for m in self.vector_store.memories.values()
            if m.memory_type == MemoryType.CONCEPT.value
        ]

        if len(concept_memories) < 2:
            return []

        contents = [m.content for m in concept_memories]
        embeddings = model.encode(contents)

        clusters = []
        assigned = set()

        for i, mem in enumerate(concept_memories):
            if mem.id in assigned:
                continue

            cluster = [mem.content]
            assigned.add(mem.id)

            for j, other_mem in enumerate(concept_memories[i+1:], i+1):
                if other_mem.id in assigned:
                    continue

                from numpy import dot
                from numpy.linalg import norm
                sim = dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]) + 1e-8)

                if sim >= min_similarity:
                    cluster.append(other_mem.content)
                    assigned.add(other_mem.id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters


semantic_engine = SemanticSearchEngine()


def add_semantic_memory(
    content: str,
    memory_type: MemoryType,
    metadata: Dict = None,
    tags: List[str] = None,
    related_alpha_id: Optional[str] = None,
    importance: float = 0.5
) -> str:
    """Add a semantic memory."""
    return semantic_engine.vector_store.add_memory(
        content, memory_type, metadata, tags, related_alpha_id, importance
    )


def search_semantic(query: str, k: int = 5, memory_types: List[MemoryType] = None) -> List[Dict]:
    """Semantic search across memories."""
    return semantic_engine.semantic_search(query, k=k, memory_types=memory_types)


def find_similar_alphas(expression: str, concepts: List[str], k: int = 5) -> List[Dict]:
    """Find similar alphas."""
    return semantic_engine.find_similar_alphas(expression, concepts, k)


def get_vector_memory_stats() -> Dict:
    """Get vector memory statistics."""
    return semantic_engine.vector_store.get_stats()


def detect_duplicate_experiment(content: str) -> List[Dict]:
    """Detect duplicate experiments."""
    return semantic_engine.detect_duplicates(content)