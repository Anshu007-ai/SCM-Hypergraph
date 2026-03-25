"""
Loss Functions for HT-HGNN

This module contains loss functions for the Hypergraph Temporal Heterogeneous Graph Neural Network,
including the NT-Xent contrastive self-supervised learning loss with temperature parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    This is the standard InfoNCE contrastive loss used in self-supervised learning,
    particularly in SimCLR and other contrastive methods.

    Args:
        z1: Augmented view embeddings of shape [N, D]
        z2: Augmented view embeddings of shape [N, D]
        temperature: Temperature parameter (tau) for scaling similarities

    Returns:
        Scalar NT-Xent contrastive loss

    Implementation Details:
        - L2 normalize both embeddings
        - Concatenate into [2N, D] representation
        - Compute cosine similarity matrix [2N, 2N]
        - Mask diagonal (self-similarity) with -inf
        - Positive pairs: (i, i+N) and (i+N, i)
        - Apply cross-entropy loss with temperature scaling
    """

    device = z1.device
    batch_size = z1.shape[0]

    # L2 normalize embeddings
    z1_norm = F.normalize(z1, p=2, dim=1)  # [N, D]
    z2_norm = F.normalize(z2, p=2, dim=1)  # [N, D]

    # Concatenate normalized embeddings
    z = torch.cat([z1_norm, z2_norm], dim=0)  # [2N, D]

    # Compute cosine similarity matrix [2N, 2N]
    sim_matrix = torch.mm(z, z.T) / temperature  # [2N, 2N]

    # Mask out diagonal (self-similarities) with -inf
    diag_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(diag_mask, float('-inf'))

    # Create positive pair labels
    # For sample i in first half: positive is i+N
    # For sample i in second half: positive is i-N
    positive_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),  # [N, 2N)
        torch.arange(0, batch_size, device=device)                # [0, N)
    ])

    # Compute NT-Xent loss using cross-entropy
    # Each row of sim_matrix is a logit distribution over negative + 1 positive
    loss = F.cross_entropy(sim_matrix, positive_indices)

    return loss


class ContrastiveMultiTaskLoss(nn.Module):
    """
    Combined loss function that includes both multi-task supervised losses
    and NT-Xent contrastive self-supervised loss.

    Combines:
    1. Price prediction (MSE)
    2. Change forecast (MSE)
    3. Critical node identification (BCE)
    4. NT-Xent contrastive loss (optional, controlled by ssl_weight)
    """

    def __init__(self,
                 weight_price: float = 1.0,
                 weight_change: float = 0.5,
                 weight_criticality: float = 0.3,
                 ssl_weight: float = 0.1,
                 ssl_temperature: float = 0.1):
        super().__init__()

        self.weight_price = weight_price
        self.weight_change = weight_change
        self.weight_criticality = weight_criticality
        self.ssl_weight = ssl_weight
        self.ssl_temperature = ssl_temperature

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid for AMP compatibility

    def forward(self,
                price_pred: torch.Tensor,
                price_target: torch.Tensor,
                change_pred: torch.Tensor,
                change_target: torch.Tensor,
                criticality_pred: torch.Tensor,
                criticality_target: torch.Tensor,
                z1: torch.Tensor = None,
                z2: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-task + contrastive loss

        Args:
            price_pred: Price predictions [N, 1]
            price_target: Price targets [N, 1]
            change_pred: Change predictions [N, 1]
            change_target: Change targets [N, 1]
            criticality_pred: Criticality logits [N, 1]
            criticality_target: Criticality targets [N, 1]
            z1: First augmented view embeddings [N, D] (optional)
            z2: Second augmented view embeddings [N, D] (optional)

        Returns:
            Dictionary with individual and total losses
        """

        # Supervised multi-task losses
        loss_price = self.mse_loss(price_pred, price_target)
        loss_change = self.mse_loss(change_pred, change_target)
        loss_criticality = self.bce_loss(criticality_pred, criticality_target)

        # Supervised loss component
        supervised_loss = (self.weight_price * loss_price +
                          self.weight_change * loss_change +
                          self.weight_criticality * loss_criticality)

        # Self-supervised contrastive loss (optional)
        ssl_loss = torch.tensor(0.0, device=price_pred.device)
        if z1 is not None and z2 is not None and self.ssl_weight > 0:
            ssl_loss = nt_xent_loss(z1, z2, self.ssl_temperature)

        # Total loss
        total_loss = supervised_loss + self.ssl_weight * ssl_loss

        return {
            'total_loss': total_loss,
            'loss_price': loss_price.item(),
            'loss_change': loss_change.item(),
            'loss_criticality': loss_criticality.item(),
            'ssl_loss': ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else ssl_loss,
            'supervised_loss': supervised_loss.item()
        }


class MultiTaskLoss(nn.Module):
    """
    Legacy multi-task learning loss for backward compatibility.

    Combines three supervised tasks:
    1. Price prediction (MSE)
    2. Change forecast (MSE)
    3. Critical node identification (Binary cross-entropy)
    """

    def __init__(self, weight_price: float = 1.0, weight_change: float = 0.5,
                 weight_criticality: float = 0.3):
        super().__init__()
        self.weight_price = weight_price
        self.weight_change = weight_change
        self.weight_criticality = weight_criticality

        self.mse_loss = nn.MSELoss()
        # Use BCEWithLogitsLoss for AMP compatibility (includes sigmoid)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                price_pred: torch.Tensor,
                price_target: torch.Tensor,
                change_pred: torch.Tensor,
                change_target: torch.Tensor,
                criticality_pred: torch.Tensor,
                criticality_target: torch.Tensor) -> Dict:
        """
        Compute weighted multi-task loss
        """

        loss_price = self.mse_loss(price_pred, price_target)
        loss_change = self.mse_loss(change_pred, change_target)
        loss_criticality = self.bce_loss(criticality_pred, criticality_target)

        total_loss = (self.weight_price * loss_price +
                     self.weight_change * loss_change +
                     self.weight_criticality * loss_criticality)

        return {
            'total_loss': total_loss,
            'loss_price': loss_price.item(),
            'loss_change': loss_change.item(),
            'loss_criticality': loss_criticality.item()
        }


if __name__ == "__main__":
    # Test NT-Xent loss implementation
    print("=" * 60)
    print("Testing NT-Xent Loss Implementation")
    print("=" * 60)

    # Create test data
    batch_size = 32
    embed_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random embeddings for two augmented views
    z1 = torch.randn(batch_size, embed_dim, device=device)
    z2 = torch.randn(batch_size, embed_dim, device=device)

    # Test different temperature values
    temperatures = [0.05, 0.1, 0.2, 0.3, 0.5]

    print(f"Batch size: {batch_size}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Device: {device}")
    print()

    for temp in temperatures:
        loss = nt_xent_loss(z1, z2, temp)
        print(f"Temperature tau={temp:.2f}: Loss={loss.item():.4f}")

    print()
    print("[OK] NT-Xent implementation test complete")

    # Test combined loss
    print("\n" + "=" * 60)
    print("Testing Combined Loss Function")
    print("=" * 60)

    loss_fn = ContrastiveMultiTaskLoss(ssl_weight=0.1, ssl_temperature=0.1)

    # Dummy supervised targets
    price_pred = torch.randn(batch_size, 1, device=device)
    price_target = torch.randn(batch_size, 1, device=device)
    change_pred = torch.randn(batch_size, 1, device=device)
    change_target = torch.randn(batch_size, 1, device=device)
    criticality_pred = torch.randn(batch_size, 1, device=device)
    criticality_target = torch.randint(0, 2, (batch_size, 1), device=device).float()

    loss_dict = loss_fn(
        price_pred, price_target,
        change_pred, change_target,
        criticality_pred, criticality_target,
        z1, z2
    )

    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Price loss: {loss_dict['loss_price']:.4f}")
    print(f"Change loss: {loss_dict['loss_change']:.4f}")
    print(f"Criticality loss: {loss_dict['loss_criticality']:.4f}")
    print(f"SSL loss: {loss_dict['ssl_loss']:.4f}")

    print("\n[OK] Combined loss function test complete")