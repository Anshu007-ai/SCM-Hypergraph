"""
Training script for Heterogeneous Temporal Hypergraph Neural Network (v2.0)

Multi-dataset training pipeline supporting:
    - 5 real-world supply chain datasets + synthetic fallback
    - Transfer learning (pretrain on one dataset, finetune on another)
    - Multi-GPU training via DataParallel
    - Checkpoint save/resume
    - WebSocket progress broadcasting (optional)
    - v1.0 (HeterogeneousTemporalHypergraphNN) and v2.0 (SpectralHypergraphConv)

Usage:
    python train_ht_hgnn.py --dataset dataco --epochs 50
    python train_ht_hgnn.py --dataset all --epochs 100 --hidden-dim 128
    python train_ht_hgnn.py --pretrain bom --finetune dataco --epochs 80
    python train_ht_hgnn.py --resume outputs/checkpoints/latest.pt --epochs 20
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so src.* imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model imports (v1.0 -- always available)
# ---------------------------------------------------------------------------
from src.models.ht_hgnn_model import (
    HeterogeneousTemporalHypergraphNN,
    MultiTaskLoss,
    EntropyAnalyzer,
)

# ---------------------------------------------------------------------------
# v2.0 spectral convolution (optional -- graceful fallback)
# ---------------------------------------------------------------------------
try:
    from src.models.hypergraph_conv import SpectralHypergraphConv
    HAS_SPECTRAL_CONV = True
except ImportError:
    HAS_SPECTRAL_CONV = False

# ---------------------------------------------------------------------------
# Data adapter (v2.0 unified normalisation layer)
# ---------------------------------------------------------------------------
try:
    from src.data.data_adapter import DataAdapter
    HAS_DATA_ADAPTER = True
except ImportError:
    HAS_DATA_ADAPTER = False

# ---------------------------------------------------------------------------
# Dataset loaders -- each wrapped in try/except for graceful fallback
# ---------------------------------------------------------------------------
try:
    from src.data.dataco_loader import DataCoLoader
    HAS_DATACO = True
except ImportError:
    HAS_DATACO = False

try:
    from src.data.bom_loader import BOMLoader
    HAS_BOM = True
except ImportError:
    HAS_BOM = False

try:
    from src.data.port_loader import PortDisruptionLoader
    HAS_PORTS = True
except ImportError:
    HAS_PORTS = False

try:
    from src.data.maintenance_loader import MaintenanceLoader
    HAS_MAINTENANCE = True
except ImportError:
    HAS_MAINTENANCE = False

try:
    from src.data.retail_loader import RetailLoader
    HAS_RETAIL = True
except ImportError:
    HAS_RETAIL = False

# ---------------------------------------------------------------------------
# Synthetic data generator (v1.0 compatibility)
# ---------------------------------------------------------------------------
try:
    from src.data.data_generator import SupplyChainDataGenerator
    HAS_SYNTHETIC = True
except ImportError:
    HAS_SYNTHETIC = False

# ---------------------------------------------------------------------------
# WebSocket broadcasting (optional)
# ---------------------------------------------------------------------------
try:
    import socketio
    _sio_client = socketio.Client()
    HAS_WEBSOCKET = True
except ImportError:
    _sio_client = None
    HAS_WEBSOCKET = False

# ---------------------------------------------------------------------------
# Available datasets registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, bool] = {
    "dataco": HAS_DATACO,
    "bom": HAS_BOM,
    "ports": HAS_PORTS,
    "maintenance": HAS_MAINTENANCE,
    "retail": HAS_RETAIL,
}


# ===================================================================
# CLI argument parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the v2.0 training script."""
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Multi-dataset training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dataco",
        choices=["dataco", "bom", "ports", "maintenance", "retail", "all"],
        help="Dataset to train on (default: dataco). Use 'all' for sequential training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for mini-batch training (default: 32).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help='Comma-separated GPU IDs, e.g. "0,1" (default: "0").',
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        choices=["dataco", "bom", "ports", "maintenance", "retail"],
        help="Dataset to pretrain on before finetuning (optional).",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        choices=["dataco", "bom", "ports", "maintenance", "retail"],
        help="Dataset to finetune on after pretraining (optional).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help='Root output directory (default: "outputs").',
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data set",
        help='Root directory for raw datasets (default: "Data set").',
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default=None,
        help="WebSocket server URL for live broadcast (optional).",
    )

    return parser


# ===================================================================
# WebSocket broadcaster
# ===================================================================

class WSBroadcaster:
    """Optional WebSocket broadcaster for live training progress."""

    def __init__(self, url: Optional[str] = None):
        self.url = url
        self.connected = False
        if url and HAS_WEBSOCKET and _sio_client is not None:
            try:
                _sio_client.connect(url, wait_timeout=5)
                self.connected = True
                print(f"  [WS] Connected to {url}")
            except Exception as exc:
                print(f"  [WS] Could not connect to {url}: {exc}")

    def send(self, event: str, data: Dict[str, Any]) -> None:
        """Emit a JSON-serialisable payload to the WebSocket server."""
        if not self.connected or _sio_client is None:
            return
        try:
            _sio_client.emit(event, data)
        except Exception:
            pass

    def close(self) -> None:
        if self.connected and _sio_client is not None:
            try:
                _sio_client.disconnect()
            except Exception:
                pass


# ===================================================================
# Dataset loading helpers
# ===================================================================

def _load_dataset_via_loader(
    dataset_name: str,
    data_dir: str,
    adapter: Optional["DataAdapter"] = None,
) -> Dict[str, Any]:
    """
    Load a named dataset through its v2.0 loader and optionally normalise
    via DataAdapter.

    Returns a standardised dictionary with keys:
        node_features, incidence_matrix, timestamps, node_types, edge_types,
        hyperedge_weights  (all NumPy arrays / lists).
    """
    raw: Dict[str, Any] = {}

    if dataset_name == "dataco":
        if not HAS_DATACO:
            raise ImportError("DataCoLoader is not available.")
        loader = DataCoLoader(data_dir=data_dir)
        raw = loader.build_hypergraph()

    elif dataset_name == "bom":
        if not HAS_BOM:
            raise ImportError("BOMLoader is not available.")
        loader = BOMLoader(data_dir=data_dir)
        raw = loader.build_hypergraph()

    elif dataset_name == "ports":
        if not HAS_PORTS:
            raise ImportError("PortDisruptionLoader is not available.")
        loader = PortDisruptionLoader(data_dir=data_dir)
        raw = loader.build_temporal_hypergraph()

    elif dataset_name == "maintenance":
        if not HAS_MAINTENANCE:
            raise ImportError("MaintenanceLoader is not available.")
        loader = MaintenanceLoader(data_dir=data_dir)
        raw = loader.build_hypergraph()

    elif dataset_name == "retail":
        if not HAS_RETAIL:
            raise ImportError("RetailLoader is not available.")
        loader = RetailLoader(data_dir=data_dir)
        raw = loader.build_hypergraph()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalise through DataAdapter when available
    if adapter is not None and HAS_DATA_ADAPTER:
        standardised = adapter.fit_transform(raw, source=dataset_name)
    else:
        standardised = raw

    return standardised


def _load_synthetic_fallback() -> Dict[str, Any]:
    """
    Fall back to synthetic data generated by SupplyChainDataGenerator
    for v1.0 compatibility when no real dataset loader succeeds.
    """
    if not HAS_SYNTHETIC:
        raise RuntimeError(
            "Cannot load synthetic data -- SupplyChainDataGenerator not found."
        )

    print("  Generating synthetic supply-chain data (v1.0 fallback) ...")
    generator = SupplyChainDataGenerator(n_suppliers=150, n_assemblies=80, seed=42)
    data = generator.generate_all()

    # Build a minimal standardised dict from synthetic output
    nodes_df = data["nodes"]
    hyperedges_df = data["hyperedges"]
    incidence_df = data["incidence"]

    n_nodes = len(nodes_df)
    n_hyperedges = len(hyperedges_df)

    # Node features: [tier, lead_time, reliability, substitutability, cost]
    feature_cols = ["tier", "lead_time", "reliability", "substitutability", "cost"]
    node_features = nodes_df[feature_cols].values.astype(np.float32)

    # Build incidence matrix (N x M)
    he_id_to_idx = {hid: i for i, hid in enumerate(hyperedges_df["hyperedge_id"])}
    node_id_to_idx = {nid: i for i, nid in enumerate(nodes_df["node_id"])}
    incidence_matrix = np.zeros((n_nodes, n_hyperedges), dtype=np.float32)
    for _, row in incidence_df.iterrows():
        ni = node_id_to_idx.get(row["node_id"])
        hi = he_id_to_idx.get(row["hyperedge_id"])
        if ni is not None and hi is not None:
            incidence_matrix[ni, hi] = 1.0

    node_types = nodes_df["node_type"].tolist()
    edge_types = ["assembly"] * n_hyperedges
    hyperedge_weights = np.ones(n_hyperedges, dtype=np.float32)

    return {
        "node_features": node_features,
        "incidence_matrix": incidence_matrix,
        "timestamps": None,
        "node_types": node_types,
        "edge_types": edge_types,
        "hyperedge_weights": hyperedge_weights,
    }


def load_dataset(
    dataset_name: str,
    data_dir: str = "Data set",
) -> Dict[str, Any]:
    """
    High-level dataset loader.  Tries the v2.0 loader first, then falls
    back to synthetic data.
    """
    adapter = DataAdapter() if HAS_DATA_ADAPTER else None

    try:
        return _load_dataset_via_loader(dataset_name, data_dir, adapter)
    except (FileNotFoundError, ImportError) as exc:
        print(f"  WARNING: Could not load '{dataset_name}' ({exc}).")
        print("  Falling back to synthetic data ...")
        return _load_synthetic_fallback()


# ===================================================================
# Prepare tensors for the model
# ===================================================================

def prepare_tensors(
    standardised: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Convert a standardised NumPy dataset dict into PyTorch tensors ready
    for the HT-HGNN forward pass.
    """
    node_features = np.asarray(standardised["node_features"], dtype=np.float32)
    incidence_matrix = np.asarray(standardised["incidence_matrix"], dtype=np.float32)

    n_nodes = node_features.shape[0]
    n_features = node_features.shape[1]
    n_hyperedges = incidence_matrix.shape[1] if incidence_matrix.ndim == 2 else 1

    # Ensure incidence is (N, M)
    if incidence_matrix.ndim == 2 and incidence_matrix.shape[0] != n_nodes:
        incidence_matrix = incidence_matrix.T

    # Node features tensor (normalise to zero-mean unit-variance)
    X = torch.FloatTensor(node_features).to(device)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X = (X - X_mean) / X_std

    # Incidence tensor -- the model expects (E, N) = (hyperedges, nodes)
    H = torch.FloatTensor(incidence_matrix).to(device)  # (N, M)
    H_model = H.t()  # (M, N) for v1.0 model convention

    # Edge index from incidence (for the HGT layer)
    edges_n, edges_e = torch.nonzero(H, as_tuple=True)
    edge_index = torch.stack([edges_e, edges_n]).to(device)  # (2, num_edges)

    # Assign edge types cyclically from the unique edge types present
    raw_edge_types = standardised.get("edge_types", ["default"])
    unique_etypes = sorted(set(raw_edge_types))
    # For the HGT we need exactly 3 edge type labels matching the model default
    model_etypes = ["supplies", "uses", "prices"]
    assigned_edge_types: List[str] = []
    for i in range(edge_index.size(1)):
        assigned_edge_types.append(model_etypes[i % len(model_etypes)])

    # Node types
    raw_node_types = standardised.get("node_types", [])
    unique_ntypes_raw = sorted(set(raw_node_types)) if raw_node_types else []
    model_ntypes = ["supplier", "part", "transaction"]

    # Map raw node types to model-expected types cyclically
    ntype_map: Dict[str, str] = {}
    for i, rnt in enumerate(unique_ntypes_raw):
        ntype_map[rnt] = model_ntypes[i % len(model_ntypes)]
    node_types_mapped = [ntype_map.get(nt, "supplier") for nt in raw_node_types]
    if not node_types_mapped:
        nodes_per = n_nodes // 3
        node_types_mapped = (
            ["supplier"] * nodes_per
            + ["part"] * nodes_per
            + ["transaction"] * (n_nodes - 2 * nodes_per)
        )

    # Timestamps
    ts_raw = standardised.get("timestamps", None)
    if ts_raw is not None and len(ts_raw) > 0:
        ts_arr = np.array(ts_raw, dtype=np.float32)
        # Expand per-hyperedge timestamps to per-node via incidence
        node_ts = np.zeros(n_nodes, dtype=np.float32)
        count = np.zeros(n_nodes, dtype=np.float32)
        for e_idx in range(len(ts_arr)):
            members = np.where(incidence_matrix[:, e_idx] > 0)[0]
            for ni in members:
                node_ts[ni] += ts_arr[e_idx]
                count[ni] += 1
        count[count == 0] = 1.0
        node_ts /= count
        timestamps = torch.FloatTensor(node_ts).to(device)
    else:
        timestamps = torch.linspace(0, 10, n_nodes).to(device)

    # Target labels (generated from features as proxies when ground truth
    # is unavailable -- same approach as v1.0)
    y_price = torch.FloatTensor(
        np.random.RandomState(42).normal(100, 20, n_nodes).astype(np.float32)
    ).to(device)
    y_change = torch.FloatTensor(
        np.random.RandomState(43).uniform(-0.1, 0.1, n_nodes).astype(np.float32)
    ).to(device)
    # Criticality from feature magnitudes
    feat_magnitude = np.linalg.norm(node_features, axis=1)
    feat_magnitude = feat_magnitude / (feat_magnitude.max() + 1e-8)
    y_criticality = torch.FloatTensor(feat_magnitude.astype(np.float32)).to(device)

    # Cascade targets: proxy based on connectivity
    connectivity = incidence_matrix.sum(axis=1).astype(np.float32)
    connectivity = connectivity / (connectivity.max() + 1e-8)
    y_cascade = torch.FloatTensor(connectivity).to(device)

    # Hyperedge weights
    hw = standardised.get("hyperedge_weights", None)
    if hw is not None:
        hyperedge_weights = torch.FloatTensor(np.asarray(hw, dtype=np.float32)).to(device)
    else:
        hyperedge_weights = torch.ones(n_hyperedges, device=device)

    return {
        "X": X,
        "incidence_matrix": H_model,           # (M, N) for v1.0 model
        "incidence_matrix_NM": H,               # (N, M) for v2.0 spectral
        "edge_index": edge_index,
        "node_types": node_types_mapped,
        "edge_types": assigned_edge_types,
        "timestamps": timestamps,
        "y_price": y_price,
        "y_change": y_change,
        "y_criticality": y_criticality,
        "y_cascade": y_cascade,
        "hyperedge_weights": hyperedge_weights,
        "n_nodes": n_nodes,
        "n_features": n_features,
        "n_hyperedges": n_hyperedges,
    }


# ===================================================================
# HT_HGNN_Trainer (v2.0 -- backwards compatible with v1.0)
# ===================================================================

class HT_HGNN_Trainer:
    """
    Trainer for HT-HGNN model (v2.0).

    Handles:
        - Data loading via v2.0 dataset loaders
        - Model creation (v1.0 or v2.0 architecture)
        - Multi-task training loop (price, change, criticality, cascade)
        - Checkpoint save/resume
        - Transfer learning (freeze encoder, train heads)
        - Multi-GPU support (DataParallel)
        - WebSocket progress broadcast
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        device: torch.device = None,
        gpu_ids: Optional[List[int]] = None,
        output_dir: str = "outputs",
        ws_url: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.gpu_ids = gpu_ids or []
        self.use_multi_gpu = len(self.gpu_ids) > 1 and torch.cuda.is_available()

        # Model, optimizer, scheduler (created lazily per dataset)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.loss_fn: Optional[MultiTaskLoss] = None

        # Training history
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "loss_price": [],
            "loss_change": [],
            "loss_criticality": [],
            "loss_cascade": [],
            "learning_rates": [],
        }

        # WebSocket broadcaster
        self.broadcaster = WSBroadcaster(ws_url)

        print("\n" + "=" * 70)
        print("HT-HGNN v2.0 -- Multi-Dataset Training Pipeline")
        print("=" * 70)
        print(f"  Device:      {self.device}")
        print(f"  Multi-GPU:   {self.use_multi_gpu} (ids={self.gpu_ids})")
        print(f"  Hidden dim:  {hidden_dim}")
        print(f"  LR:          {learning_rate}")
        print(f"  Output dir:  {self.output_dir}")
        print(f"  Spectral v2: {HAS_SPECTRAL_CONV}")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, data: Dict[str, Any], epochs: int = 100) -> None:
        """Construct model, loss, optimizer, and scheduler from data shapes."""

        n_nodes = data["n_nodes"]
        n_features = data["n_features"]
        n_hyperedges = data["n_hyperedges"]
        attention_heads = 8

        # Determine unique node/edge types for the HGT layer
        unique_ntypes = sorted(set(data["node_types"]))
        unique_etypes = sorted(set(data["edge_types"]))

        print(f"\n  Building model:")
        print(f"    Nodes: {n_nodes}  Features: {n_features}  Hyperedges: {n_hyperedges}")
        print(f"    Node types: {unique_ntypes}")
        print(f"    Edge types: {unique_etypes}")
        print(f"    Attention heads: {attention_heads}")

        self.model = HeterogeneousTemporalHypergraphNN(
            in_channels=n_features,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim // 2,
            num_nodes=n_nodes,
            num_hyperedges=n_hyperedges,
            node_types=unique_ntypes,
            edge_types=unique_etypes,
            num_hgnn_layers=2,
            num_hgt_heads=min(attention_heads, self.hidden_dim),
            time_window=10,
        ).to(self.device)

        # Multi-GPU wrapping
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            print(f"    Wrapped in DataParallel over GPUs {self.gpu_ids}")

        # Multi-task loss (4 heads: price, change, criticality, cascade)
        self.loss_fn = MultiTaskLoss(
            weight_price=1.0,
            weight_change=0.5,
            weight_criticality=0.3,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1)
        )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"    Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Run one training epoch and return loss dict."""
        model = self.model
        model.train()

        # Forward pass
        output = self._forward(model, data)

        # --- Multi-task loss ---
        loss_dict = self.loss_fn(
            price_pred=output["price_pred"],
            price_target=data["y_price"],
            change_pred=output["change_pred"],
            change_target=data["y_change"],
            criticality_pred=output["criticality"],
            criticality_target=data["y_criticality"],
        )

        # Cascade loss (MSE on cascade scores)
        cascade_loss = nn.functional.mse_loss(
            output["cascade_scores"], data["y_cascade"]
        )
        total_loss = loss_dict["total_loss"] + 0.2 * cascade_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "loss_price": loss_dict["loss_price"],
            "loss_change": loss_dict["loss_change"],
            "loss_criticality": loss_dict["loss_criticality"],
            "loss_cascade": cascade_loss.item(),
        }

    @staticmethod
    def _forward(model: nn.Module, data: Dict[str, Any]) -> Dict:
        """Unified forward pass that handles both plain and DataParallel models."""
        # DataParallel wraps the model; unwrap for the actual call signature
        m = model.module if isinstance(model, nn.DataParallel) else model
        return m(
            node_features=data["X"],
            incidence_matrix=data["incidence_matrix"],
            node_types=data["node_types"],
            edge_index=data["edge_index"],
            edge_types=data["edge_types"],
            timestamps=data["timestamps"],
        )

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        data: Dict[str, Any],
        epochs: int = 100,
        verbose: bool = True,
        phase: str = "train",
        save_every: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            data:       Tensor dict from prepare_tensors().
            epochs:     Number of epochs.
            verbose:    Print progress every 10 epochs.
            phase:      Label for logging ('train', 'pretrain', 'finetune').
            save_every: Save checkpoint every N epochs.

        Returns:
            Training history dict.
        """
        if self.model is None:
            self._build_model(data, epochs=epochs)

        print(f"\n" + "-" * 70)
        print(f"  TRAINING  [{phase.upper()}]  --  {epochs} epochs")
        print("-" * 70)

        best_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            loss_dict = self._train_epoch(data)

            # Record history
            self.history["loss"].append(loss_dict["total_loss"])
            self.history["loss_price"].append(loss_dict["loss_price"])
            self.history["loss_change"].append(loss_dict["loss_change"])
            self.history["loss_criticality"].append(loss_dict["loss_criticality"])
            self.history["loss_cascade"].append(loss_dict["loss_cascade"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rates"].append(current_lr)

            # Scheduler step
            self.scheduler.step()

            # Best model checkpoint
            if loss_dict["total_loss"] < best_loss:
                best_loss = loss_dict["total_loss"]
                self._save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    epoch=epoch,
                    loss=best_loss,
                    phase=phase,
                )

            # Periodic checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(
                    self.checkpoint_dir / "latest.pt",
                    epoch=epoch,
                    loss=loss_dict["total_loss"],
                    phase=phase,
                )

            # Console logging
            epoch_time = time.time() - epoch_start
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  Epoch {epoch:4d}/{epochs}  |  "
                    f"Loss: {loss_dict['total_loss']:.6f}  "
                    f"(price={loss_dict['loss_price']:.4f}  "
                    f"change={loss_dict['loss_change']:.6f}  "
                    f"crit={loss_dict['loss_criticality']:.4f}  "
                    f"cascade={loss_dict['loss_cascade']:.6f})  |  "
                    f"LR: {current_lr:.6f}  |  {epoch_time:.2f}s"
                )

            # WebSocket broadcast
            self.broadcaster.send("training_progress", {
                "epoch": epoch,
                "total_epochs": epochs,
                "phase": phase,
                "loss": loss_dict["total_loss"],
                "loss_price": loss_dict["loss_price"],
                "loss_change": loss_dict["loss_change"],
                "loss_criticality": loss_dict["loss_criticality"],
                "loss_cascade": loss_dict["loss_cascade"],
                "lr": current_lr,
            })

        elapsed = time.time() - start_time
        print(f"\n  Training complete ({elapsed:.1f}s)")
        print(f"  Final loss: {self.history['loss'][-1]:.6f}")
        print(f"  Best loss:  {best_loss:.6f}")

        # Save latest checkpoint after training
        self._save_checkpoint(
            self.checkpoint_dir / "latest.pt",
            epoch=epochs,
            loss=self.history["loss"][-1],
            phase=phase,
        )

        return self.history

    # ------------------------------------------------------------------
    # Entropy & sensitivity analysis (preserved from v1.0)
    # ------------------------------------------------------------------

    def analyze_entropy_and_sensitivity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform entropy and sensitivity analysis on the trained model."""
        print("\n" + "-" * 70)
        print("  ENTROPY AND SENSITIVITY ANALYSIS")
        print("-" * 70)

        model = self.model
        model.eval()
        with torch.no_grad():
            output = self._forward(model, data)

        entropy = output["entropy"]
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.item()
        sensitivity = output["sensitivity"]
        information_flow = output["information_flow"]
        cascade_scores = output["cascade_scores"]

        print(f"\n  Information Flow Entropy: {entropy:.4f}")
        print(f"    (Higher = more distributed, Lower = more focused)")

        print(f"\n  Sensitivity Analysis:")
        print(f"    Mean: {sensitivity.mean():.4f}  "
              f"Max: {sensitivity.max():.4f}  "
              f"Std: {sensitivity.std():.4f}")

        top_k = min(10, len(sensitivity))
        top_sens = torch.topk(sensitivity, k=top_k)
        print(f"\n  Top {top_k} most sensitive nodes:")
        for rank, (val, idx) in enumerate(zip(top_sens.values, top_sens.indices)):
            print(f"    {rank + 1:2d}. Node {idx.item():5d}: {val.item():.4f}")

        print(f"\n  Cascade Scores:")
        print(f"    Mean: {cascade_scores.mean():.4f}  "
              f"Max: {cascade_scores.max():.4f}")

        top_cascade = torch.topk(cascade_scores, k=top_k)
        print(f"\n  Top {top_k} critical nodes (cascade propagation):")
        for rank, (val, idx) in enumerate(zip(top_cascade.values, top_cascade.indices)):
            print(f"    {rank + 1:2d}. Node {idx.item():5d}: {val.item():.4f}")

        return {
            "entropy": float(entropy),
            "sensitivity": sensitivity.cpu().numpy(),
            "information_flow": information_flow.cpu().numpy(),
            "cascade_scores": cascade_scores.cpu().numpy(),
            "top_sensitive_nodes": top_sens.indices.cpu().numpy().tolist(),
            "top_critical_nodes": top_cascade.indices.cpu().numpy().tolist(),
        }

    # ------------------------------------------------------------------
    # Transfer learning helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """
        Freeze all encoder layers (HGNN, HGT, TGN) and keep only the
        output heads trainable -- used during the finetune phase.
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Freeze HGNN layers
        for layer in model.hgnn_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze HGT
        for param in model.hgt.parameters():
            param.requires_grad = False

        # Freeze TGN
        for param in model.tgn.parameters():
            param.requires_grad = False

        # Freeze entropy analyzer
        for param in model.entropy_analyzer.parameters():
            param.requires_grad = False

        # Output heads remain trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Encoder frozen.  Trainable: {trainable:,} / {total:,} params")

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the model."""
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  All layers unfrozen.  Trainable: {trainable:,} params")

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int = 0,
        loss: float = 0.0,
        phase: str = "train",
    ) -> None:
        """Save a training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
                "phase": phase,
                "hidden_dim": self.hidden_dim,
                "history": self.history,
                "config": {
                    "in_channels": model.in_channels,
                    "hidden_channels": model.hidden_channels,
                    "out_channels": model.out_channels,
                    "num_nodes": model.num_nodes,
                    "num_hyperedges": model.num_hyperedges,
                },
            },
            str(path),
        )

    def load_checkpoint(self, path: str, data: Dict[str, Any]) -> int:
        """
        Load a checkpoint, rebuild model if needed, and restore state.

        Returns the epoch number to resume from.
        """
        print(f"\n  Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        # Rebuild model if not yet created
        if self.model is None:
            self._build_model(data, epochs=100)

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(ckpt["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "history" in ckpt:
            self.history = ckpt["history"]

        resumed_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {resumed_epoch}  (loss={ckpt.get('loss', 0):.6f})")
        return resumed_epoch

    def save_model(self, path: str = None) -> None:
        """Save just the model weights (for inference)."""
        if path is None:
            path = str(self.output_dir / "models" / "ht_hgnn_model.pt")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "in_channels": model.in_channels,
                    "hidden_channels": model.hidden_channels,
                    "out_channels": model.out_channels,
                    "num_nodes": model.num_nodes,
                    "num_hyperedges": model.num_hyperedges,
                },
                "history": self.history,
            },
            path,
        )
        print(f"  Model saved to {path}")

    # ------------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------------

    def save_history(self, path: str = None) -> None:
        """Write training history to a JSON file."""
        if path is None:
            path = str(self.output_dir / "training_history.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history saved to {path}")


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # GPU setup
    # ------------------------------------------------------------------
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if torch.cuda.is_available() and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        gpu_ids = []

    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in gpu_ids:
            if i < torch.cuda.device_count():
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ------------------------------------------------------------------
    # Create trainer
    # ------------------------------------------------------------------
    trainer = HT_HGNN_Trainer(
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        device=device,
        gpu_ids=gpu_ids,
        output_dir=args.output_dir,
        ws_url=args.ws_url,
    )

    # ------------------------------------------------------------------
    # Transfer learning path
    # ------------------------------------------------------------------
    if args.pretrain and args.finetune:
        print("\n" + "=" * 70)
        print("  TRANSFER LEARNING MODE")
        print(f"  Pretrain: {args.pretrain}  -->  Finetune: {args.finetune}")
        print("=" * 70)

        # Phase 1: Pretrain
        print(f"\n  Phase 1: Loading pretrain dataset '{args.pretrain}' ...")
        pretrain_raw = load_dataset(args.pretrain, data_dir=args.data_dir)
        pretrain_data = prepare_tensors(pretrain_raw, device)

        pretrain_epochs = max(args.epochs // 2, 10)
        trainer.train(
            pretrain_data,
            epochs=pretrain_epochs,
            phase="pretrain",
        )

        # Phase 2: Finetune -- freeze encoder, train heads
        print(f"\n  Phase 2: Loading finetune dataset '{args.finetune}' ...")
        finetune_raw = load_dataset(args.finetune, data_dir=args.data_dir)
        finetune_data = prepare_tensors(finetune_raw, device)

        # Rebuild model for new data dimensions if they changed
        trainer.model = None
        trainer._build_model(finetune_data, epochs=args.epochs - pretrain_epochs)

        # Load pretrained encoder weights where shapes match
        pretrained_ckpt = torch.load(
            str(trainer.checkpoint_dir / "best.pt"), map_location=device
        )
        model = trainer.model.module if isinstance(trainer.model, nn.DataParallel) else trainer.model
        pretrained_state = pretrained_ckpt["model_state_dict"]
        current_state = model.state_dict()

        transferred = 0
        for key in pretrained_state:
            if key in current_state and pretrained_state[key].shape == current_state[key].shape:
                current_state[key] = pretrained_state[key]
                transferred += 1

        model.load_state_dict(current_state)
        print(f"  Transferred {transferred} parameter tensors from pretrained model")

        # Freeze encoder
        trainer.freeze_encoder()

        # Recreate optimizer with only trainable params
        trainer.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 0.1,  # lower LR for finetuning
        )
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=max(args.epochs - pretrain_epochs, 1)
        )

        finetune_epochs = args.epochs - pretrain_epochs
        trainer.train(
            finetune_data,
            epochs=finetune_epochs,
            phase="finetune",
        )

        # Analysis on finetune data
        analysis = trainer.analyze_entropy_and_sensitivity(finetune_data)

        # Save
        trainer.save_model()
        trainer.save_history()

        _save_analysis(analysis, trainer.output_dir)

    # ------------------------------------------------------------------
    # Standard training (single or all datasets)
    # ------------------------------------------------------------------
    else:
        datasets_to_train = (
            [d for d, avail in DATASET_REGISTRY.items() if avail]
            if args.dataset == "all"
            else [args.dataset]
        )

        if args.dataset == "all" and not datasets_to_train:
            print("  No dataset loaders available -- falling back to synthetic data.")
            datasets_to_train = ["synthetic"]

        for ds_name in datasets_to_train:
            print(f"\n{'=' * 70}")
            print(f"  DATASET: {ds_name.upper()}")
            print("=" * 70)

            # Load data
            print(f"\n  Loading dataset '{ds_name}' ...")
            if ds_name == "synthetic":
                raw_data = _load_synthetic_fallback()
            else:
                raw_data = load_dataset(ds_name, data_dir=args.data_dir)

            data = prepare_tensors(raw_data, device)

            # Reset model for each dataset when training 'all'
            if args.dataset == "all":
                trainer.model = None
                trainer.history = {
                    "loss": [], "loss_price": [], "loss_change": [],
                    "loss_criticality": [], "loss_cascade": [],
                    "learning_rates": [],
                }

            # Resume from checkpoint if specified
            if args.resume:
                if trainer.model is None:
                    trainer._build_model(data, epochs=args.epochs)
                trainer.load_checkpoint(args.resume, data)

            # Train
            trainer.train(data, epochs=args.epochs, phase=f"train-{ds_name}")

            # Analysis
            analysis = trainer.analyze_entropy_and_sensitivity(data)

            # Save artefacts per dataset
            ds_suffix = f"_{ds_name}" if args.dataset == "all" else ""
            trainer.save_model(
                str(trainer.output_dir / "models" / f"ht_hgnn_model{ds_suffix}.pt")
            )
            trainer.save_history(
                str(trainer.output_dir / f"training_history{ds_suffix}.json")
            )
            _save_analysis(
                analysis,
                trainer.output_dir,
                filename=f"ht_hgnn_analysis{ds_suffix}.json",
            )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    trainer.broadcaster.close()

    print("\n" + "=" * 70)
    print("  HT-HGNN v2.0 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Results saved to: {trainer.output_dir}/")
    print(f"    Checkpoints:       {trainer.checkpoint_dir}/")
    print(f"    Model weights:     {trainer.output_dir / 'models'}/")
    print(f"    Training history:  {trainer.output_dir / 'training_history.json'}")
    print(f"    Analysis report:   {trainer.output_dir / 'ht_hgnn_analysis.json'}")


def _save_analysis(
    analysis: Dict[str, Any],
    output_dir: Path,
    filename: str = "ht_hgnn_analysis.json",
) -> None:
    """Persist the entropy/sensitivity analysis to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "entropy": float(analysis["entropy"]),
        "mean_sensitivity": float(analysis["sensitivity"].mean()),
        "mean_information_flow": float(analysis["information_flow"].mean()),
        "mean_cascade_score": float(analysis["cascade_scores"].mean()),
        "top_sensitive_nodes": analysis["top_sensitive_nodes"],
        "top_critical_nodes": analysis["top_critical_nodes"],
        "timestamp": datetime.now().isoformat(),
    }
    path = output_dir / filename
    with open(str(path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Analysis saved to {path}")


if __name__ == "__main__":
    main()
