"""
Alpha Lineage System
Tracks alpha ancestry, experiment evolution, parameter changes, and branching history.
"""

import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib


@dataclass
class AlphaAncestry:
    """Alpha derivation chain."""
    alpha_id: str
    root_id: Optional[str]
    parents: List[str]
    children: List[str]
    created_at: str
    modified_at: str
    modification_count: int


@dataclass
class Modification:
    """A single modification to an alpha."""
    modification_id: str
    timestamp: str
    modification_type: str  # parameter_change, operator_substitution, factor_evolution, branch
    before_state: Dict
    after_state: Dict
    reason: Optional[str]
    performance_delta: Optional[float]


@dataclass
class ExperimentBranch:
    """A branch in the alpha experiment tree."""
    branch_id: str
    parent_id: str
    created_at: str
    name: str
    description: str
    status: str  # active, merged, abandoned
    alphas: List[str]


@dataclass
class LineageGraph:
    """Complete lineage for an alpha."""
    alpha_id: str
    ancestry: AlphaAncestry
    modifications: List[Modification]
    branches: List[str]
    related_alphas: Dict[str, List[str]]  # correlated, replaced, improved


class LineageTracker:
    """Track alpha lineage and evolution."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "lineage_data.json"

        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        """Load lineage data."""
        self.ancestry: Dict[str, AlphaAncestry] = {}
        self.modifications: Dict[str, List[Modification]] = defaultdict(list)
        self.branches: Dict[str, ExperimentBranch] = {}
        self.branch_membership: Dict[str, str] = {}  # alpha_id -> branch_id

        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)

                    for a_id, a_data in data.get("ancestry", {}).items():
                        self.ancestry[a_id] = AlphaAncestry(**a_data)

                    for m_list in data.get("modifications", {}).values():
                        for m_data in m_list:
                            self.modifications[m_data.get("modification_id", "")].append(
                                Modification(**m_data)
                            )

                    for b_id, b_data in data.get("branches", {}).items():
                        self.branches[b_id] = ExperimentBranch(**b_data)

                    self.branch_membership = data.get("branch_membership", {})
            except Exception:
                pass

    def _save_data(self):
        """Save lineage data."""
        data = {
            "ancestry": {k: asdict(v) for k, v in self.ancestry.items()},
            "modifications": {
                k: [asdict(m) for m in v]
                for k, v in self.modifications.items()
            },
            "branches": {k: asdict(v) for k, v in self.branches.items()},
            "branch_membership": self.branch_membership
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_alpha(
        self,
        alpha_id: str,
        parent_id: Optional[str] = None,
        root_id: Optional[str] = None,
        branch_id: Optional[str] = None
    ) -> AlphaAncestry:
        """Register a new alpha with lineage tracking."""
        now = datetime.now().isoformat()

        if parent_id and parent_id in self.ancestry:
            parent = self.ancestry[parent_id]
            if root_id is None:
                root_id = parent.root_id or parent_id
            children = list(parent.children)
            if alpha_id not in children:
                children.append(alpha_id)
            parent.children = children

        ancestry = AlphaAncestry(
            alpha_id=alpha_id,
            root_id=root_id or alpha_id,
            parents=[parent_id] if parent_id else [],
            children=[],
            created_at=now,
            modified_at=now,
            modification_count=0
        )

        self.ancestry[alpha_id] = ancestry

        if branch_id:
            self.branch_membership[alpha_id] = branch_id

        self._save_data()
        return ancestry

    def record_modification(
        self,
        alpha_id: str,
        modification_type: str,
        before_state: Dict,
        after_state: Dict,
        reason: Optional[str] = None,
        performance_delta: Optional[float] = None
    ) -> Modification:
        """Record a modification to an alpha."""
        mod_id = hashlib.md5(
            f"{alpha_id}_{modification_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        modification = Modification(
            modification_id=mod_id,
            timestamp=datetime.now().isoformat(),
            modification_type=modification_type,
            before_state=before_state,
            after_state=after_state,
            reason=reason,
            performance_delta=performance_delta
        )

        self.modifications[alpha_id].append(modification)

        if alpha_id in self.ancestry:
            self.ancestry[alpha_id].modified_at = modification.timestamp
            self.ancestry[alpha_id].modification_count += 1

        self._save_data()
        return modification

    def create_branch(
        self,
        branch_name: str,
        parent_alpha_id: str,
        description: str = ""
    ) -> ExperimentBranch:
        """Create a new experiment branch."""
        branch_id = f"branch_{hashlib.md5(branch_name.encode()).hexdigest()[:8]}"

        branch = ExperimentBranch(
            branch_id=branch_id,
            parent_id=parent_alpha_id,
            created_at=datetime.now().isoformat(),
            name=branch_name,
            description=description,
            status="active",
            alphas=[parent_alpha_id]
        )

        self.branches[branch_id] = branch
        self.branch_membership[parent_alpha_id] = branch_id

        self._save_data()
        return branch

    def add_to_branch(self, alpha_id: str, branch_id: str) -> None:
        """Add an alpha to a branch."""
        if branch_id in self.branches:
            self.branches[branch_id].alphas.append(alpha_id)
            self.branch_membership[alpha_id] = branch_id
            self._save_data()

    def merge_branch(self, branch_id: str, target_branch_id: str) -> None:
        """Merge one branch into another."""
        if branch_id in self.branches and target_branch_id in self.branches:
            self.branches[branch_id].status = "merged"
            for alpha_id in self.branches[branch_id].alphas:
                self.branch_membership[alpha_id] = target_branch_id
                if alpha_id not in self.branches[target_branch_id].alphas:
                    self.branches[target_branch_id].alphas.append(alpha_id)
            self._save_data()

    def get_ancestry(self, alpha_id: str) -> Optional[AlphaAncestry]:
        """Get ancestry for an alpha."""
        return self.ancestry.get(alpha_id)

    def get_lineage(self, alpha_id: str) -> Optional[LineageGraph]:
        """Get complete lineage for an alpha."""
        if alpha_id not in self.ancestry:
            return None

        ancestry = self.ancestry[alpha_id]

        root_id = ancestry.root_id
        root_ancestry = self.ancestry.get(root_id)

        all_children = []
        if root_ancestry:
            to_visit = list(root_ancestry.children)
            while to_visit:
                child_id = to_visit.pop()
                if child_id != alpha_id:
                    all_children.append(child_id)
                child = self.ancestry.get(child_id)
                if child:
                    to_visit.extend(child.children)

        related = {
            "parents": ancestry.parents,
            "children": ancestry.children,
            "siblings": self._get_siblings(alpha_id),
            "cousins": all_children[:10]
        }

        return LineageGraph(
            alpha_id=alpha_id,
            ancestry=ancestry,
            modifications=self.modifications.get(alpha_id, []),
            branches=list(self.branches.keys()),
            related_alphas=related
        )

    def _get_siblings(self, alpha_id: str) -> List[str]:
        """Get sibling alphas (same parents)."""
        if alpha_id not in self.ancestry:
            return []

        ancestry = self.ancestry[alpha_id]
        siblings = []

        for parent_id in ancestry.parents:
            parent = self.ancestry.get(parent_id)
            if parent:
                for child_id in parent.children:
                    if child_id != alpha_id:
                        siblings.append(child_id)

        return siblings

    def compare_alphas(
        self,
        alpha_1: str,
        alpha_2: str
    ) -> Dict:
        """Compare two alphas in the same lineage."""
        lineage_1 = self.get_lineage(alpha_1)
        lineage_2 = self.get_lineage(alpha_2)

        if not lineage_1 or not lineage_2:
            return {"error": "One or both alphas not found in lineage"}

        common_ancestor = None
        if lineage_1.ancestry.root_id == lineage_2.ancestry.root_id:
            common_ancestor = lineage_1.ancestry.root_id

        mod_count_1 = len(self.modifications.get(alpha_1, []))
        mod_count_2 = len(self.modifications.get(alpha_2, []))

        return {
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "common_ancestor": common_ancestor,
            "same_lineage": lineage_1.ancestry.root_id == lineage_2.ancestry.root_id,
            "modifications_1": mod_count_1,
            "modifications_2": mod_count_2,
            "depth_1": self._get_depth(alpha_1),
            "depth_2": self._get_depth(alpha_2)
        }

    def _get_depth(self, alpha_id: str) -> int:
        """Get depth in the lineage tree."""
        if alpha_id not in self.ancestry:
            return 0

        ancestry = self.ancestry[alpha_id]
        depth = 0

        current = ancestry
        while current.parents:
            parent_id = current.parents[0]
            parent = self.ancestry.get(parent_id)
            if not parent:
                break
            depth += 1
            current = parent

        return depth

    def get_tree(self, root_alpha_id: str, max_depth: int = 5) -> Dict:
        """Get the full tree starting from a root alpha."""
        if root_alpha_id not in self.ancestry:
            return {"error": "Alpha not found"}

        def build_node(alpha_id: str, depth: int) -> Dict:
            if depth > max_depth:
                return {"id": alpha_id, "truncated": True}

            ancestry = self.ancestry.get(alpha_id)
            if not ancestry:
                return {"id": alpha_id}

            node = {
                "id": alpha_id,
                "created_at": ancestry.created_at,
                "modification_count": ancestry.modification_count,
                "children": []
            }

            for child_id in ancestry.children:
                child_node = build_node(child_id, depth + 1)
                node["children"].append(child_node)

            return node

        return build_node(root_alpha_id, 0)

    def get_statistics(self) -> Dict:
        """Get lineage tracking statistics."""
        total_alphas = len(self.ancestry)
        total_modifications = sum(len(m) for m in self.modifications.values())
        active_branches = len([b for b in self.branches.values() if b.status == "active"])

        roots = [a_id for a_id, a in self.ancestry.items() if not a.parents]

        return {
            "total_tracked_alphas": total_alphas,
            "root_alphas": len(roots),
            "total_modifications": total_modifications,
            "active_branches": active_branches,
            "merged_branches": len([b for b in self.branches.values() if b.status == "merged"]),
            "avg_modifications_per_alpha": round(total_modifications / max(1, total_alphas), 2)
        }


lineage_tracker = LineageTracker()


def register_alpha(
    alpha_id: str,
    parent_id: Optional[str] = None,
    root_id: Optional[str] = None,
    branch_id: Optional[str] = None
) -> Dict:
    """Register a new alpha with lineage."""
    return asdict(lineage_tracker.register_alpha(alpha_id, parent_id, root_id, branch_id))


def record_modification(
    alpha_id: str,
    modification_type: str,
    before_state: Dict,
    after_state: Dict,
    reason: Optional[str] = None,
    performance_delta: Optional[float] = None
) -> Dict:
    """Record a modification."""
    return asdict(lineage_tracker.record_modification(
        alpha_id, modification_type, before_state, after_state, reason, performance_delta
    ))


def get_lineage(alpha_id: str) -> Optional[Dict]:
    """Get complete lineage for an alpha."""
    lineage = lineage_tracker.get_lineage(alpha_id)
    if lineage:
        return {
            "alpha_id": lineage.alpha_id,
            "ancestry": asdict(lineage.ancestry),
            "modifications": [asdict(m) for m in lineage.modifications],
            "related_alphas": lineage.related_alphas
        }
    return None


def compare_lineage(alpha_1: str, alpha_2: str) -> Dict:
    """Compare two alphas."""
    return lineage_tracker.compare_alphas(alpha_1, alpha_2)


def get_tree(root_alpha_id: str) -> Dict:
    """Get full experiment tree."""
    return lineage_tracker.get_tree(root_alpha_id)


def get_lineage_stats() -> Dict:
    """Get lineage statistics."""
    return lineage_tracker.get_statistics()


def create_experiment_branch(name: str, parent_alpha_id: str, description: str = "") -> Dict:
    """Create a new experiment branch."""
    return asdict(lineage_tracker.create_branch(name, parent_alpha_id, description))