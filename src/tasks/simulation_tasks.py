"""
Asynchronous Simulation Celery Tasks

Provides Celery task wrappers around the core simulation and explainability
engines so that long-running computations (cascade simulation, stress testing,
model explanations) can be executed asynchronously by a pool of workers.

Each task:
  1. Loads or reconstructs the necessary engine objects.
  2. Delegates to the engine's own methods.
  3. Returns a JSON-serializable result dictionary.

Usage example (from application code)::

    from src.tasks.simulation_tasks import run_cascade_simulation
    result = run_cascade_simulation.delay(
        shock_nodes=["supplier_3", "supplier_7"],
        shock_magnitude=0.8,
        params={"propagation_decay": 0.5},
    )
    print(result.get(timeout=120))
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hypergraph():
    """
    Load or reconstruct the current Hypergraph instance.

    In a production setting this would deserialize from a cache (Redis, disk,
    or database).  For now it tries to import and call a conventional loader
    function, falling back to an empty Hypergraph.
    """
    try:
        from src.data.data_generator import SupplyChainDataGenerator
        generator = SupplyChainDataGenerator()
        data = generator.generate_dataset()
        from src.hypergraph.hypergraph import Hypergraph
        hg = Hypergraph.from_dataframes(
            nodes_df=data["nodes_df"],
            hyperedges_df=data["hyperedges_df"],
            incidence_df=data["incidence_df"],
            echelon_df=data.get("echelon_df"),
        )
        return hg
    except Exception as exc:
        logger.warning("Could not load hypergraph via data generator: %s", exc)
        from src.hypergraph.hypergraph import Hypergraph
        return Hypergraph()


def _load_failure_simulator(hypergraph):
    """Instantiate a FailureSimulator with a minimal feature/label stub."""
    try:
        import pandas as pd
        from src.evaluation.validation import FailureSimulator
        empty_df = pd.DataFrame()
        return FailureSimulator(
            hypergraph=hypergraph,
            features_df=empty_df,
            labels_df=empty_df,
            model=None,
        )
    except Exception as exc:
        logger.warning("Could not load FailureSimulator: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Task: Cascade Simulation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="src.tasks.simulation_tasks.run_cascade_simulation",
    max_retries=2,
    default_retry_delay=30,
)
def run_cascade_simulation(
    self,
    shock_nodes: List[str],
    shock_magnitude: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a cascade simulation starting from one or more shock nodes.

    Parameters
    ----------
    shock_nodes : list of str
        Node IDs where the disruption originates.
    shock_magnitude : float
        Severity of the initial shock in [0, 1].
    params : dict or None
        Additional simulation parameters (e.g. ``propagation_decay``).

    Returns
    -------
    dict
        Simulation results including affected nodes, cascade depth, and
        per-node impact scores.
    """
    task_id = self.request.id
    logger.info("Cascade simulation started (task=%s, nodes=%s)", task_id, shock_nodes)
    start = time.time()

    params = params or {}

    try:
        hg = _load_hypergraph()
        simulator = _load_failure_simulator(hg)

        all_affected: Dict[str, Any] = {}
        cascade_details: Dict[str, Any] = {}

        for node_id in shock_nodes:
            if simulator is not None:
                result = simulator.simulate_supplier_failure(
                    supplier_id=node_id,
                    failure_severity=shock_magnitude,
                )
            else:
                # Lightweight fallback using hypergraph directly
                affected_he = hg.node_to_hyperedges.get(node_id, set())
                downstream = set()
                for he_id in affected_he:
                    downstream |= hg.get_downstream_hyperedges(he_id)
                result = {
                    "failed_supplier": node_id,
                    "directly_affected_hyperedges": list(affected_he),
                    "n_affected": len(affected_he),
                    "cascade_details": {
                        he: {"downstream_ids": list(downstream)}
                        for he in affected_he
                    },
                }

            cascade_details[node_id] = result
            if "directly_affected_hyperedges" in result:
                for he in result["directly_affected_hyperedges"]:
                    all_affected[he] = all_affected.get(he, 0) + 1

        elapsed = time.time() - start
        logger.info("Cascade simulation done (task=%s, %.2fs)", task_id, elapsed)

        return {
            "task_id": task_id,
            "status": "success",
            "shock_nodes": shock_nodes,
            "shock_magnitude": shock_magnitude,
            "params": params,
            "total_affected_hyperedges": len(all_affected),
            "affected_hyperedges": all_affected,
            "per_node_cascades": cascade_details,
            "elapsed_seconds": round(elapsed, 3),
        }

    except Exception as exc:
        logger.exception("Cascade simulation failed (task=%s)", task_id)
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Task: Stress Test
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="src.tasks.simulation_tasks.run_stress_test",
    max_retries=1,
    default_retry_delay=60,
)
def run_stress_test(
    self,
    n_scenarios: int = 10,
    strategy: str = "random",
) -> Dict[str, Any]:
    """
    Run a stress test composed of multiple failure scenarios.

    Parameters
    ----------
    n_scenarios : int
        Number of failure scenarios to simulate.
    strategy : str
        Node selection strategy: ``'random'``, ``'high_risk'``, or
        ``'targeted'``.

    Returns
    -------
    dict
        Aggregated stress-test results including vulnerability statistics.
    """
    task_id = self.request.id
    logger.info(
        "Stress test started (task=%s, n=%d, strategy=%s)",
        task_id, n_scenarios, strategy,
    )
    start = time.time()

    try:
        hg = _load_hypergraph()
        simulator = _load_failure_simulator(hg)

        if simulator is not None:
            stress_result = simulator.run_stress_test(n_failures=n_scenarios)
        else:
            # Minimal fallback
            import numpy as np
            node_ids = list(hg.nodes.keys())
            n = min(n_scenarios, len(node_ids))
            if n == 0:
                stress_result = {
                    "n_failures_tested": 0,
                    "total_hyperedges_at_risk": 0,
                    "avg_cascade_size": 0,
                    "max_cascade_size": 0,
                    "vulnerability_rate": 0.0,
                }
            else:
                chosen = np.random.choice(node_ids, size=n, replace=False)
                impacts = []
                for nid in chosen:
                    affected = hg.node_to_hyperedges.get(nid, set())
                    impacts.append(len(affected))
                stress_result = {
                    "n_failures_tested": n,
                    "total_hyperedges_at_risk": int(sum(impacts)),
                    "avg_cascade_size": float(np.mean(impacts)) if impacts else 0,
                    "max_cascade_size": int(max(impacts)) if impacts else 0,
                    "vulnerability_rate": (
                        sum(impacts) / max(len(hg.hyperedges), 1)
                    ),
                }

        elapsed = time.time() - start
        logger.info("Stress test done (task=%s, %.2fs)", task_id, elapsed)

        return {
            "task_id": task_id,
            "status": "success",
            "n_scenarios": n_scenarios,
            "strategy": strategy,
            "results": stress_result,
            "elapsed_seconds": round(elapsed, 3),
        }

    except Exception as exc:
        logger.exception("Stress test failed (task=%s)", task_id)
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Task: Explanation
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="src.tasks.simulation_tasks.run_explanation",
    max_retries=2,
    default_retry_delay=30,
)
def run_explanation(
    self,
    node_ids: List[str],
    prediction_type: str = "risk",
) -> Dict[str, Any]:
    """
    Generate model explanations for a set of nodes.

    Wraps the explainability engine (when available) to produce
    feature-attribution or structural-importance explanations.

    Parameters
    ----------
    node_ids : list of str
        Node IDs to explain.
    prediction_type : str
        Type of prediction to explain: ``'risk'``, ``'cascade'``, or
        ``'ranking'``.

    Returns
    -------
    dict
        Per-node explanation payloads.
    """
    task_id = self.request.id
    logger.info(
        "Explanation started (task=%s, nodes=%s, type=%s)",
        task_id, node_ids, prediction_type,
    )
    start = time.time()

    try:
        hg = _load_hypergraph()

        # Attempt to use a dedicated explainability engine
        explanations: Dict[str, Any] = {}
        engine_available = False

        try:
            # Placeholder: import your explainability engine once implemented
            # from src.explainability.engine import ExplainabilityEngine
            # engine = ExplainabilityEngine(hg)
            # engine_available = True
            pass
        except ImportError:
            pass

        for nid in node_ids:
            if nid not in hg.nodes:
                explanations[nid] = {"error": f"Node '{nid}' not found in hypergraph"}
                continue

            node = hg.nodes[nid]
            memberships = list(hg.node_to_hyperedges.get(nid, set()))

            if engine_available:
                # explanations[nid] = engine.explain(nid, prediction_type)
                pass
            else:
                # Structural explanation fallback
                explanations[nid] = {
                    "node_id": nid,
                    "prediction_type": prediction_type,
                    "node_type": node.node_type,
                    "tier": node.tier,
                    "reliability": float(node.reliability),
                    "substitutability": float(node.substitutability),
                    "lead_time": node.lead_time,
                    "n_hyperedge_memberships": len(memberships),
                    "hyperedge_memberships": memberships,
                    "structural_importance": (
                        len(memberships) / max(len(hg.hyperedges), 1)
                    ),
                    "note": (
                        "Structural explanation only; dedicated explainability "
                        "engine not yet available."
                    ),
                }

        elapsed = time.time() - start
        logger.info("Explanation done (task=%s, %.2fs)", task_id, elapsed)

        return {
            "task_id": task_id,
            "status": "success",
            "prediction_type": prediction_type,
            "explanations": explanations,
            "elapsed_seconds": round(elapsed, 3),
        }

    except Exception as exc:
        logger.exception("Explanation failed (task=%s)", task_id)
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    print("Simulation tasks module loaded.")
    print("Registered tasks:")
    print("  - run_cascade_simulation")
    print("  - run_stress_test")
    print("  - run_explanation")
    print(
        "\nTo invoke synchronously (testing only):\n"
        "  result = run_cascade_simulation(['supplier_0'], 0.8)\n"
        "  print(result)"
    )
