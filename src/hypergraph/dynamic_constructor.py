"""
Temporal Co-Occurrence Hyperedge Mining

Dynamically constructs hyperedges from event logs using temporal co-occurrence
patterns. This extends the static hypergraph with data-driven hyperedge discovery:

Algorithm: Temporal Co-Occurrence Hyperedge Mining
Input:  Event log E, window size W, support threshold tau
Output: Dynamic hyperedge set H_dyn

For each window [t, t+W]:
  1. Find all node sets S that co-appear in >= tau events
  2. Filter S by |S| >= 3
  3. Add S as new hyperedge with weight = co-occurrence count
  4. Merge overlapping hyperedges using Jaccard similarity > 0.7
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass, field

from src.hypergraph.hypergraph import Hypergraph, HypergraphEdge


@dataclass
class DynamicHyperedge:
    """Represents a dynamically discovered hyperedge from event mining."""
    hyperedge_id: str
    nodes: Set[str] = field(default_factory=set)
    weight: float = 0.0
    window_start: Optional[pd.Timestamp] = None
    window_end: Optional[pd.Timestamp] = None
    support_count: int = 0

    def __repr__(self) -> str:
        return (
            f"DynamicHyperedge(id={self.hyperedge_id}, "
            f"nodes={self.nodes}, weight={self.weight}, "
            f"support={self.support_count})"
        )


class DynamicHyperedgeConstructor:
    """
    Discovers and constructs hyperedges from temporal event co-occurrence patterns.

    Given a time-stamped event log where nodes (e.g., suppliers, components)
    appear together in events (e.g., disruptions, shipments, quality incidents),
    this class mines frequent co-occurrence sets within sliding time windows
    and converts them into hyperedges that augment the existing hypergraph.

    Parameters
    ----------
    window_size : int
        Size of the sliding time window in days (default: 30).
    support_threshold : int
        Minimum number of co-occurrences within a window for a node set to
        qualify as a hyperedge (default: 3).
    min_hyperedge_size : int
        Minimum number of nodes required in a hyperedge (default: 3).
    jaccard_threshold : float
        Jaccard similarity threshold above which two candidate hyperedges
        are merged (default: 0.7).
    """

    def __init__(
        self,
        window_size: int = 30,
        support_threshold: int = 3,
        min_hyperedge_size: int = 3,
        jaccard_threshold: float = 0.7,
    ):
        self.window_size = window_size
        self.support_threshold = support_threshold
        self.min_hyperedge_size = min_hyperedge_size
        self.jaccard_threshold = jaccard_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mine_from_events(
        self,
        event_log: pd.DataFrame,
        time_col: str = "timestamp",
        node_col: str = "node_id",
        event_id_col: str = "event_id",
    ) -> List[DynamicHyperedge]:
        """
        Mine dynamic hyperedges from a temporal event log.

        The event log must contain at minimum a timestamp column, a node
        identifier column, and an event identifier that groups co-occurring
        nodes within a single event.

        Parameters
        ----------
        event_log : pd.DataFrame
            Event log with columns for timestamps, node IDs, and event IDs.
        time_col : str
            Name of the timestamp column (default: ``'timestamp'``).
        node_col : str
            Name of the node identifier column (default: ``'node_id'``).
        event_id_col : str
            Name of the event identifier column (default: ``'event_id'``).
            If the column does not exist, each unique timestamp is treated
            as a separate event.

        Returns
        -------
        list of DynamicHyperedge
            Discovered hyperedges, already merged for overlap.
        """
        df = event_log.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        # If no explicit event_id column, synthesize one per unique timestamp
        if event_id_col not in df.columns:
            df[event_id_col] = df[time_col].astype(str)

        all_candidates: List[DynamicHyperedge] = []
        counter = 0

        # Sliding window
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        window_delta = pd.Timedelta(days=self.window_size)
        current_start = min_time

        while current_start <= max_time:
            current_end = current_start + window_delta
            window_df = df[
                (df[time_col] >= current_start) & (df[time_col] < current_end)
            ]

            if not window_df.empty:
                candidates = self._mine_window(
                    window_df, node_col, event_id_col,
                    current_start, current_end,
                    counter,
                )
                all_candidates.extend(candidates)
                counter += len(candidates)

            # Slide forward by half the window for overlap
            current_start += window_delta / 2

        # Deduplicate identical node sets across windows (keep highest weight)
        deduped = self._deduplicate(all_candidates)

        # Merge overlapping hyperedges
        merged = self.merge_overlapping(deduped)

        return merged

    def merge_overlapping(
        self, hyperedges: List[DynamicHyperedge]
    ) -> List[DynamicHyperedge]:
        """
        Merge hyperedges whose node sets have Jaccard similarity exceeding
        the configured threshold.

        Uses single-linkage agglomerative merging: when two hyperedges are
        merged, the resulting union is compared against remaining hyperedges
        until no further merges are possible.

        Parameters
        ----------
        hyperedges : list of DynamicHyperedge
            Candidate hyperedges to merge.

        Returns
        -------
        list of DynamicHyperedge
            Merged hyperedge list (may be shorter than the input).
        """
        if not hyperedges:
            return []

        merged: List[DynamicHyperedge] = list(hyperedges)
        changed = True

        while changed:
            changed = False
            new_merged: List[DynamicHyperedge] = []
            consumed: Set[int] = set()

            for i in range(len(merged)):
                if i in consumed:
                    continue
                current = merged[i]

                for j in range(i + 1, len(merged)):
                    if j in consumed:
                        continue

                    sim = self.jaccard_similarity(current.nodes, merged[j].nodes)
                    if sim > self.jaccard_threshold:
                        # Merge j into current
                        current = DynamicHyperedge(
                            hyperedge_id=current.hyperedge_id,
                            nodes=current.nodes | merged[j].nodes,
                            weight=current.weight + merged[j].weight,
                            window_start=min(
                                filter(None, [current.window_start, merged[j].window_start])
                            ) if current.window_start or merged[j].window_start else None,
                            window_end=max(
                                filter(None, [current.window_end, merged[j].window_end])
                            ) if current.window_end or merged[j].window_end else None,
                            support_count=current.support_count + merged[j].support_count,
                        )
                        consumed.add(j)
                        changed = True

                new_merged.append(current)
                consumed.add(i)

            merged = new_merged

        return merged

    @staticmethod
    def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """
        Compute the Jaccard similarity between two sets.

        Parameters
        ----------
        set_a : set of str
        set_b : set of str

        Returns
        -------
        float
            Jaccard index in [0, 1]. Returns 0.0 if both sets are empty.
        """
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def update_hypergraph(
        self,
        hypergraph: Hypergraph,
        new_hyperedges: List[DynamicHyperedge],
        prefix: str = "dyn",
    ) -> Hypergraph:
        """
        Integrate dynamically discovered hyperedges into an existing
        Hypergraph instance.

        Each :class:`DynamicHyperedge` is converted to a
        :class:`HypergraphEdge` and added to the hypergraph. Only nodes
        that already exist in the hypergraph are included; unknown nodes
        are silently dropped.

        Parameters
        ----------
        hypergraph : Hypergraph
            The existing static hypergraph to augment.
        new_hyperedges : list of DynamicHyperedge
            Dynamically mined hyperedges.
        prefix : str
            Prefix to prepend to dynamic hyperedge IDs to avoid collisions
            with existing static hyperedge IDs (default: ``'dyn'``).

        Returns
        -------
        Hypergraph
            The same hypergraph object, now containing the new hyperedges.
        """
        existing_ids = set(hypergraph.hyperedges.keys())

        for dyn_he in new_hyperedges:
            # Ensure unique ID
            he_id = f"{prefix}_{dyn_he.hyperedge_id}"
            if he_id in existing_ids:
                he_id = f"{prefix}_{dyn_he.hyperedge_id}_{id(dyn_he)}"

            # Keep only nodes known to the hypergraph
            valid_nodes = {n for n in dyn_he.nodes if n in hypergraph.nodes}
            if len(valid_nodes) < self.min_hyperedge_size:
                continue

            edge = HypergraphEdge(
                hyperedge_id=he_id,
                nodes=valid_nodes,
                bom_weight=dyn_he.weight,
                tolerance=0.0,
                critical_path=0.0,
                tier_level=0,
            )
            hypergraph.add_hyperedge(edge)
            existing_ids.add(he_id)

        return hypergraph

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mine_window(
        self,
        window_df: pd.DataFrame,
        node_col: str,
        event_id_col: str,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        id_offset: int,
    ) -> List[DynamicHyperedge]:
        """Mine co-occurring node sets within a single time window."""
        # Group nodes by event to find co-occurrence sets
        event_groups = window_df.groupby(event_id_col)[node_col].apply(set)

        # Count pair-wise and set-wise co-occurrences
        cooccurrence: Dict[frozenset, int] = defaultdict(int)

        for _event_id, node_set in event_groups.items():
            node_list = sorted(node_set)
            # Enumerate all subsets of size >= min_hyperedge_size
            for size in range(self.min_hyperedge_size, len(node_list) + 1):
                for combo in combinations(node_list, size):
                    cooccurrence[frozenset(combo)] += 1

        # Filter by support threshold
        candidates: List[DynamicHyperedge] = []
        counter = 0
        for node_set_frozen, count in cooccurrence.items():
            if count >= self.support_threshold:
                candidates.append(
                    DynamicHyperedge(
                        hyperedge_id=f"dyn_{id_offset + counter}",
                        nodes=set(node_set_frozen),
                        weight=float(count),
                        window_start=window_start,
                        window_end=window_end,
                        support_count=count,
                    )
                )
                counter += 1

        # Remove subsets: if A subset of B and both pass threshold, keep only B
        candidates = self._remove_subsets(candidates)

        return candidates

    @staticmethod
    def _remove_subsets(candidates: List[DynamicHyperedge]) -> List[DynamicHyperedge]:
        """Remove candidates that are strict subsets of another candidate."""
        if not candidates:
            return candidates

        # Sort by node set size descending so larger sets are processed first
        sorted_cands = sorted(candidates, key=lambda c: len(c.nodes), reverse=True)
        kept: List[DynamicHyperedge] = []

        for cand in sorted_cands:
            is_subset = False
            for existing in kept:
                if cand.nodes <= existing.nodes:
                    is_subset = True
                    break
            if not is_subset:
                kept.append(cand)

        return kept

    @staticmethod
    def _deduplicate(
        hyperedges: List[DynamicHyperedge],
    ) -> List[DynamicHyperedge]:
        """
        Deduplicate hyperedges with identical node sets across overlapping
        windows, keeping the one with the highest weight.
        """
        best: Dict[frozenset, DynamicHyperedge] = {}
        for he in hyperedges:
            key = frozenset(he.nodes)
            if key not in best or he.weight > best[key].weight:
                best[key] = he
        return list(best.values())


if __name__ == "__main__":
    # ---- Demo: mine dynamic hyperedges from a synthetic event log ----
    np.random.seed(42)

    # Create a synthetic event log
    n_events = 50
    nodes = [f"supplier_{i}" for i in range(10)]
    events = []
    for eid in range(n_events):
        ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=np.random.randint(0, 90))
        involved = np.random.choice(nodes, size=np.random.randint(3, 6), replace=False)
        for node in involved:
            events.append({"event_id": f"evt_{eid}", "timestamp": ts, "node_id": node})

    event_log = pd.DataFrame(events)
    print(f"Synthetic event log: {len(event_log)} rows, {n_events} events")

    constructor = DynamicHyperedgeConstructor(
        window_size=30,
        support_threshold=2,
        min_hyperedge_size=3,
        jaccard_threshold=0.7,
    )

    dynamic_hes = constructor.mine_from_events(event_log)
    print(f"\nDiscovered {len(dynamic_hes)} dynamic hyperedges:")
    for he in dynamic_hes[:5]:
        print(f"  {he}")

    print("\nDynamic constructor module ready for integration")
