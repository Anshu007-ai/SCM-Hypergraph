"""
Cascade Risk Scoring Head (v2.0)

Implements multi-step cascade risk propagation through a hypergraph
structure.  The core idea is that disruptions propagate from individual
nodes through shared hyperedges according to:

    P(e disrupted at t+1) = 1 - prod_{v in e} (1 - P(v disrupted at t))^alpha

where alpha is a learnable *interaction strength* parameter that controls
how strongly co-membership in a hyperedge amplifies disruption probability.

The model iterates this propagation for ``num_cascade_steps`` rounds,
producing per-node cascade risk scores that quantify each node's exposure
to cascading failures.

Also includes ``DynamicHyperedgeWeightLearner`` which dynamically
computes hyperedge weights conditioned on the current node embeddings
and a time embedding:

    w(e, t) = MLP([mu(e), sigma(e), delta(e), t_embedding])

where mu, sigma, delta are the mean, standard deviation, and max-min
range of node embeddings within hyperedge e, respectively.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ======================================================================
# Dynamic Hyperedge Weight Learner
# ======================================================================

class DynamicHyperedgeWeightLearner(nn.Module):
    """Learn time-dependent hyperedge weights from node statistics.

    For each hyperedge e at time step t, the weight is:
        w(e, t) = MLP([mu(e), sigma(e), delta(e), t_embedding])

    where:
        mu(e)    = mean of node embeddings in e
        sigma(e) = std of node embeddings in e
        delta(e) = max - min of node embeddings in e (range)
        t_embedding = sinusoidal encoding of the time step

    Args:
        hidden_dim:     Dimensionality of node embeddings.
        time_embed_dim: Dimensionality of the time embedding.
        mlp_hidden:     Hidden layer size in the weight MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        time_embed_dim: int = 32,
        mlp_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding projection
        self.time_proj = nn.Linear(1, time_embed_dim)

        # Input = 3 * hidden_dim (mu, sigma, delta) + time_embed_dim
        mlp_input_dim = 3 * hidden_dim + time_embed_dim

        self.weight_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
            nn.Sigmoid(),  # weights in [0, 1]
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        for module in self.weight_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        incidence_matrix: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dynamic hyperedge weights.

        Args:
            node_embeddings: Node features of shape (N, hidden_dim).
            incidence_matrix: Binary incidence matrix of shape (N, M)
                              where H[v, e] = 1 iff vertex v in hyperedge e.
            time_step:       Scalar or shape-(1,) tensor representing
                             the current time step.  Defaults to 0.

        Returns:
            weights: Hyperedge weight vector of shape (M,) in [0, 1].
        """
        N, M = incidence_matrix.shape
        device = node_embeddings.device

        if time_step is None:
            time_step = torch.zeros(1, device=device)
        time_step = time_step.float().reshape(1, 1)  # (1, 1)

        # Time embedding
        t_embed = F.relu(self.time_proj(time_step))  # (1, time_embed_dim)
        t_embed = t_embed.expand(M, -1)              # (M, time_embed_dim)

        # Compute per-hyperedge node statistics
        # Mask for which nodes belong to each hyperedge
        # incidence_matrix^T: (M, N)
        H_t = incidence_matrix.t()  # (M, N)
        degrees = H_t.sum(dim=1, keepdim=True).clamp(min=1)  # (M, 1)

        # Weighted sum of node embeddings per hyperedge
        # (M, N) @ (N, hidden_dim) -> (M, hidden_dim)
        he_sum = torch.matmul(H_t, node_embeddings)
        mu = he_sum / degrees  # mean: (M, hidden_dim)

        # Variance / std -- computed via E[X^2] - E[X]^2
        he_sq_sum = torch.matmul(H_t, node_embeddings ** 2)
        variance = (he_sq_sum / degrees) - mu ** 2
        sigma = variance.clamp(min=1e-8).sqrt()  # (M, hidden_dim)

        # Range (max - min per hyperedge)
        # For efficiency we approximate with a soft approach:
        #   max ~ mu + 2*sigma, min ~ mu - 2*sigma  =>  delta ~ 4*sigma
        # But we can also compute it exactly for moderate-size graphs.
        # Here we do exact computation.
        delta = torch.zeros(M, self.hidden_dim, device=device)
        for e_idx in range(M):
            member_mask = H_t[e_idx] > 0  # (N,)
            if member_mask.any():
                members = node_embeddings[member_mask]  # (k, hidden_dim)
                e_max = members.max(dim=0).values
                e_min = members.min(dim=0).values
                delta[e_idx] = e_max - e_min

        # Concatenate statistics + time embedding
        mlp_input = torch.cat([mu, sigma, delta, t_embed], dim=-1)  # (M, 3*H + T)

        # Predict weights
        weights = self.weight_mlp(mlp_input).squeeze(-1)  # (M,)
        return weights


# ======================================================================
# Cascade Risk Head
# ======================================================================

class CascadeRiskHead(nn.Module):
    """Multi-step cascade risk propagation through hypergraph structure.

    Computes per-node disruption risk by iteratively propagating
    disruption probabilities through shared hyperedges:

        P(e disrupted at t+1) = 1 - prod_{v in e} (1 - P(v disrupted at t))^alpha
        P(v disrupted at t+1) = max_{e containing v} P(e disrupted at t+1)

    The interaction strength alpha is learned from data (initialised to 1).

    Args:
        hidden_dim:        Node embedding dimensionality.
        num_cascade_steps: Number of propagation iterations.
        cascade_threshold: Probability threshold above which a node is
                           considered ``at risk'' (used for reporting only;
                           does not affect differentiable outputs).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_cascade_steps: int = 12,
        cascade_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_cascade_steps = num_cascade_steps
        self.cascade_threshold = cascade_threshold

        # ---- Initial risk estimator ----------------------------------
        # Maps node embeddings to initial disruption probability
        self.risk_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # ---- Learnable interaction strength alpha --------------------
        # Initialised to 1.0 (neutral); learn from data
        self.log_alpha = nn.Parameter(torch.zeros(1))  # alpha = exp(log_alpha)

        # ---- Per-step damping factor ---------------------------------
        # Prevents unbounded risk accumulation
        self.damping = nn.Parameter(torch.tensor(0.9))

        # ---- Dynamic hyperedge weight learner ------------------------
        self.weight_learner = DynamicHyperedgeWeightLearner(
            hidden_dim=hidden_dim
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialise all learnable parameters."""
        for module in self.risk_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        nn.init.zeros_(self.log_alpha)
        nn.init.constant_(self.damping, 0.9)
        self.weight_learner.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        """Interaction strength (always positive)."""
        return self.log_alpha.exp()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        incidence_matrix: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run multi-step cascade risk propagation.

        Args:
            node_embeddings: Node features of shape (N, hidden_dim).
            incidence_matrix: Binary incidence matrix of shape (N, M).
            time_step:       Optional current time step for dynamic
                             hyperedge weight computation.

        Returns:
            cascade_risk_scores: Per-node cascade risk probabilities
                                 of shape (N,) in [0, 1].
        """
        N, M = incidence_matrix.shape
        device = node_embeddings.device

        # ---- Step 0: Initial node risk from embeddings ---------------
        p_node = self.risk_mlp(node_embeddings).squeeze(-1)  # (N,)

        # ---- Learn dynamic hyperedge weights -------------------------
        he_weights = self.weight_learner(
            node_embeddings, incidence_matrix, time_step
        )  # (M,)

        # ---- Cascade propagation -------------------------------------
        alpha = self.alpha                  # scalar > 0
        damping = torch.sigmoid(self.damping)  # in (0, 1)

        # Pre-compute H^T for node->hyperedge aggregation
        H = incidence_matrix               # (N, M)
        H_t = H.t()                        # (M, N)

        # Track risk over cascade steps (for analysis / debugging)
        risk_history = [p_node]

        for step in range(self.num_cascade_steps):
            # --- Node -> Hyperedge propagation ---
            # P(e disrupted) = 1 - prod_{v in e} (1 - P(v))^alpha
            #
            # In log-space for numerical stability:
            #   log(1 - P(e)) = sum_{v in e} alpha * log(1 - P(v))
            #   P(e) = 1 - exp(sum_{v in e} alpha * log(1 - P(v)))

            log_survival_node = torch.log(
                (1.0 - p_node).clamp(min=1e-8)
            )  # (N,)

            # Sum log-survival of member nodes per hyperedge
            # (M, N) @ (N,) -> (M,)
            log_survival_he = torch.matmul(H_t, log_survival_node)
            log_survival_he = alpha * log_survival_he

            # Apply hyperedge weights -- higher weight = stronger coupling
            log_survival_he = log_survival_he * he_weights

            p_he = 1.0 - torch.exp(log_survival_he.clamp(max=0))  # (M,)
            p_he = p_he.clamp(0.0, 1.0)

            # --- Hyperedge -> Node propagation ---
            # P(v disrupted at t+1) = max over hyperedges containing v
            # We use a soft-max approximation for differentiability:
            #   p_node_new = 1 - prod_{e containing v} (1 - P(e))

            log_survival_he_for_node = torch.log(
                (1.0 - p_he).clamp(min=1e-8)
            )  # (M,)

            # (N, M) @ (M,) -> (N,)
            log_survival_from_he = torch.matmul(H, log_survival_he_for_node)

            p_node_from_he = 1.0 - torch.exp(
                log_survival_from_he.clamp(max=0)
            )  # (N,)
            p_node_from_he = p_node_from_he.clamp(0.0, 1.0)

            # Blend with previous risk (damped update)
            p_node = damping * p_node_from_he + (1.0 - damping) * p_node
            p_node = p_node.clamp(0.0, 1.0)

            risk_history.append(p_node)

        # Final cascade risk scores
        cascade_risk_scores = p_node  # (N,)

        return cascade_risk_scores

    def get_risk_summary(
        self, cascade_risk_scores: torch.Tensor
    ) -> dict:
        """Generate a human-readable risk summary.

        Args:
            cascade_risk_scores: Output of ``forward()``.

        Returns:
            Dictionary with risk statistics.
        """
        scores = cascade_risk_scores.detach()
        at_risk = (scores > self.cascade_threshold).sum().item()
        return {
            "mean_risk": scores.mean().item(),
            "max_risk": scores.max().item(),
            "min_risk": scores.min().item(),
            "std_risk": scores.std().item(),
            "nodes_at_risk": int(at_risk),
            "total_nodes": int(scores.numel()),
            "risk_ratio": at_risk / max(scores.numel(), 1),
            "alpha": self.alpha.item(),
            "threshold": self.cascade_threshold,
        }


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CascadeRiskHead -- Module Info & Smoke Test")
    print("=" * 60)

    hidden_dim = 64
    num_nodes = 30
    num_hyperedges = 10
    num_cascade_steps = 12

    model = CascadeRiskHead(
        hidden_dim=hidden_dim,
        num_cascade_steps=num_cascade_steps,
        cascade_threshold=0.5,
    )
    print(f"\nModel:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Synthetic data
    node_emb = torch.randn(num_nodes, hidden_dim)

    # Random incidence matrix (each hyperedge contains 2-6 nodes)
    H = torch.zeros(num_nodes, num_hyperedges)
    for e in range(num_hyperedges):
        size = torch.randint(2, 7, (1,)).item()
        members = torch.randperm(num_nodes)[:size]
        H[members, e] = 1.0

    time_step = torch.tensor([5.0])

    print(f"\nIncidence matrix shape: {H.shape}")
    print(f"Hyperedge sizes:        {H.sum(dim=0).int().tolist()}")

    # Forward pass
    risk_scores = model(node_emb, H, time_step)
    print(f"\nInput  node_embeddings shape: {node_emb.shape}")
    print(f"Output cascade_risk shape:    {risk_scores.shape}")
    print(f"Risk score range:             [{risk_scores.min().item():.4f}, "
          f"{risk_scores.max().item():.4f}]")

    # Risk summary
    summary = model.get_risk_summary(risk_scores)
    print(f"\nRisk summary:")
    for key, val in summary.items():
        print(f"  {key:20s}: {val}")

    # Dynamic weight learner standalone test
    print(f"\n--- DynamicHyperedgeWeightLearner ---")
    weights = model.weight_learner(node_emb, H, time_step)
    print(f"Learned hyperedge weights: {weights.detach().tolist()}")

    # Gradient check
    loss = risk_scores.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"\nGradient flow OK: {grad_ok}")
    print(f"Learned alpha:    {model.alpha.item():.4f}")

    print("\nSmoke test passed.")
