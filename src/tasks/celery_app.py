"""
Celery Application Configuration

Configures the Celery distributed task queue for running long-running
supply-chain simulations asynchronously (cascade simulation, stress
testing, explainability).

Broker:  Redis at localhost:6379/0
Backend: Redis at localhost:6379/1
"""

from celery import Celery

# ---------------------------------------------------------------------------
# Create the Celery application
# ---------------------------------------------------------------------------

celery_app = Celery(
    "sc_hypergraph",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=[
        "src.tasks.simulation_tasks",
    ],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Time zones
    timezone="UTC",
    enable_utc=True,

    # Result expiry (24 hours)
    result_expires=86400,

    # Prefetch multiplier (1 = fair scheduling for long tasks)
    worker_prefetch_multiplier=1,

    # Acknowledge late so tasks survive worker crashes
    task_acks_late=True,

    # Reject tasks on worker shutdown so they can be redelivered
    task_reject_on_worker_lost=True,

    # Soft/hard time limits (seconds)
    task_soft_time_limit=600,
    task_time_limit=900,

    # Task routing
    task_routes={
        "src.tasks.simulation_tasks.run_cascade_simulation": {
            "queue": "simulations",
        },
        "src.tasks.simulation_tasks.run_stress_test": {
            "queue": "simulations",
        },
        "src.tasks.simulation_tasks.run_explanation": {
            "queue": "explanations",
        },
    },

    # Default queue for anything not explicitly routed
    task_default_queue="default",
)


# ---------------------------------------------------------------------------
# Health-check task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="src.tasks.celery_app.health_check")
def health_check(self):
    """
    Lightweight health-check task used by monitoring endpoints.

    Returns
    -------
    dict
        Status payload including the worker hostname and task ID.
    """
    return {
        "status": "ok",
        "worker": self.request.hostname,
        "task_id": self.request.id,
    }


if __name__ == "__main__":
    print("Celery app configured.")
    print(f"  Broker:  {celery_app.conf.broker_url}")
    print(f"  Backend: {celery_app.conf.result_backend}")
    print(f"  Queues:  default, simulations, explanations")
    print(
        "\nTo start a worker:\n"
        "  celery -A src.tasks.celery_app worker --loglevel=info -Q default,simulations,explanations"
    )
