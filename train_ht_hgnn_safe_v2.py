"""
Production-Ready HT-HGNN Training with PyTorch CUDA
Features:
- Robust CUDA support with fallback to CPU
- Comprehensive error handling
- Memory management and validation
- Safe data loading with verification
- Production-grade logging
- Checkpoint saving
- Resource monitoring
- MOO Ablation Study support for 2-stage pipeline

MOO Integration:
- Stage 1: Multi-Objective Optimization over aviation hypergraph
- Stage 2: HT-HGNN neural network with MOO transfer mechanisms
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import warnings
from typing import Dict, Tuple, Optional, List
import sys
import argparse
from datetime import datetime

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from src.models.ht_hgnn_model import (
    HeterogeneousTemporalHypergraphNN,
    MultiTaskLoss
)

# Import MOO ablation system
from moo_ablation import (
    MOOTransferConfig,
    CONFIG_MAP,
    CONFIG_FULL,
    build_node_features,
    get_loss_weights,
    get_cascade_targets,
    run_moo_ablation_study
)

# Import SSL loss functions
from losses import ContrastiveMultiTaskLoss


# Configure logging - UTF-8 safe format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SafeDeviceManager:
    """Manages device selection with safety checks"""
    
    @staticmethod
    def get_device() -> torch.device:
        """Safely get available device"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                device = torch.device('cuda')
                logger.info(f"[CUDA] Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"[CUDA] Device count: {torch.cuda.device_count()}")
                logger.info(f"[CUDA] Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                return device
        except RuntimeError as e:
            logger.warning(f"[WARN] CUDA error: {e}")
        
        logger.info("[DEVICE] Using CPU (CUDA unavailable)")
        return torch.device('cpu')
    
    @staticmethod
    def clear_cache():
        """Safely clear GPU cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("[CUDA] Cache cleared")
        except Exception as e:
            logger.debug(f"[WARN] Cache clear warning: {e}")


class SafeDataValidator:
    """Validates input data for safety"""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor, 
        name: str, 
        expected_shape: Optional[Tuple] = None
    ) -> bool:
        """Validate tensor for NaN/Inf and shape"""
        try:
            # Check for NaN
            if torch.isnan(tensor).any():
                logger.error(f"[ERROR] {name} contains NaN values")
                return False
            
            # Check for Inf
            if torch.isinf(tensor).any():
                logger.error(f"[ERROR] {name} contains Inf values")
                return False
            
            # Check shape if expected
            if expected_shape and tensor.shape != expected_shape:
                logger.error(f"[ERROR] {name} shape mismatch. Expected {expected_shape}, got {tensor.shape}")
                return False
            
            logger.debug(f"[OK] {name} validated: shape={tensor.shape}, dtype={tensor.dtype}")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Error validating {name}: {e}")
            return False
    
    @staticmethod
    def validate_csv_exists(path: str) -> bool:
        """Check if CSV file exists and is readable"""
        try:
            csv_path = Path(path)
            if not csv_path.exists():
                logger.error(f"[ERROR] File not found: {path}")
                return False
            
            if csv_path.stat().st_size == 0:
                logger.error(f"[ERROR] File is empty: {path}")
                return False
            
            logger.debug(f"[OK] File exists: {path} ({csv_path.stat().st_size / 1024:.1f} KB)")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Error checking file {path}: {e}")
            return False


class SafeHT_HGNN_Trainer:
    """
    Production-ready trainer with comprehensive safety and MOO integration

    Features:
    - Safe device selection
    - Data validation
    - Memory monitoring
    - Checkpoint saving
    - Error recovery
    - MOO Transfer Mechanism Support (Feature/Loss/HIC)
    """

    def __init__(
        self,
        in_channels: int = 18,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_nodes: int = 1206,
        num_hyperedges: int = 36,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        checkpoint_dir: str = 'outputs/checkpoints',
        moo_config: Optional[MOOTransferConfig] = None,
        ssl_temperature: float = 0.1,
        ssl_weight: float = 0.1,
        attention_type: str = 'structural'
    ):
        """Initialize trainer with safety checks, MOO support, SSL temperature control, and attention type"""

        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device management
        self.device = SafeDeviceManager.get_device()

        # MOO Configuration
        self.moo_config = moo_config or CONFIG_FULL

        # SSL Configuration
        self.ssl_temperature = ssl_temperature
        self.ssl_weight = ssl_weight

        # Attention Configuration
        self.attention_type = attention_type

        # Configuration
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.use_amp = use_amp and self.device.type == 'cuda'

        logger.info("\n" + "="*80)
        logger.info("HT-HGNN PRODUCTION TRAINER WITH MOO INTEGRATION")
        logger.info("="*80)
        logger.info(f"Model Configuration:")
        logger.info(f"  Input channels: {in_channels}")
        logger.info(f"  Hidden channels: {hidden_channels}")
        logger.info(f"  Output channels: {out_channels}")
        logger.info(f"  Nodes: {num_nodes}")
        logger.info(f"  Hyperedges: {num_hyperedges}")
        logger.info(f"")
        logger.info(f"MOO Configuration: {self.moo_config.name}")
        logger.info(f"  Description: {self.moo_config.description}")
        logger.info(f"  Enabled mechanisms: {', '.join(self.moo_config.enabled_mechanisms) if self.moo_config.enabled_mechanisms else 'None'}")
        logger.info(f"  Expected input dim: {self.moo_config.input_dim}")
        logger.info(f"  Use MOO features: {self.moo_config.use_moo_features}")
        logger.info(f"  Use MOO loss weights: {self.moo_config.use_moo_loss_weights}")
        logger.info(f"  Use HIC targets: {self.moo_config.use_hic_targets}")
        logger.info(f"")
        logger.info(f"SSL Configuration:")
        logger.info(f"  Temperature (τ): {self.ssl_temperature}")
        logger.info(f"  SSL weight: {self.ssl_weight}")
        logger.info(f"")
        logger.info(f"Attention Configuration:")
        logger.info(f"  Aggregation type: {self.attention_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed Precision: {self.use_amp}")
        
        try:
            # Initialize model with safety
            logger.info("Initializing model...")
            self.model = HeterogeneousTemporalHypergraphNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_nodes=num_nodes,
                num_hyperedges=num_hyperedges,
                node_types=['supplier', 'part', 'transaction'],
                edge_types=['supplies', 'uses', 'prices'],
                num_hgnn_layers=2,
                num_hgt_heads=4,
                time_window=10
            ).to(self.device)
            
            # Loss function with SSL support
            self.loss_fn = ContrastiveMultiTaskLoss(
                weight_price=1.0,
                weight_change=0.5,
                weight_criticality=0.3,
                ssl_weight=self.ssl_weight,
                ssl_temperature=self.ssl_temperature
            ).to(self.device)
            
            # Optimizer with weight decay
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=1e-8
            )
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
            
            # Mixed precision training
            if self.use_amp:
                self.scaler = GradScaler()
                logger.info("[AMP] Mixed precision training enabled")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"[MODEL] Initialized with {total_params:,} parameters")
            
            # Training history
            self.history = {
                'loss': [],
                'loss_price': [],
                'loss_change': [],
                'loss_criticality': [],
                'cascade_kl': [],
                'learning_rates': []
            }
            
            self.start_epoch = 0
            
        except Exception as e:
            logger.error(f"[ERROR] Model initialization failed: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Safely load checkpoint"""
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(f"[WARN] Checkpoint not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0)
            self.history = checkpoint.get('history', self.history)
            
            logger.info(f"[CHECKPOINT] Loaded from {checkpoint_path} (epoch {self.start_epoch})")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Checkpoint loading failed: {e}")
            return False
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> bool:
        """Safely save checkpoint"""
        try:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
                'config': {
                    'in_channels': self.in_channels,
                    'hidden_channels': self.hidden_channels,
                    'out_channels': self.out_channels,
                    'num_nodes': self.num_nodes,
                    'num_hyperedges': self.num_hyperedges
                }
            }
            
            # Save latest checkpoint
            latest_path = self.checkpoint_dir / 'latest.pt'
            torch.save(checkpoint_data, latest_path)
            
            # Save best checkpoint if applicable
            if is_best:
                best_path = self.checkpoint_dir / 'best.pt'
                torch.save(checkpoint_data, best_path)
                logger.info(f"[CHECKPOINT] Best model saved (epoch {epoch})")
            
            logger.debug(f"[CHECKPOINT] Saved: epoch {epoch}")
            return True
        
        except Exception as e:
            logger.error(f"[ERROR] Checkpoint saving failed: {e}")
            return False
    
    def prepare_data(
        self,
        features_csv: str = 'outputs/datasets/features.csv',
        hci_csv: str = 'outputs/datasets/hci_labels.csv',
        incidence_csv: str = 'outputs/datasets/incidence.csv'
    ) -> Dict:
        """Safely prepare and validate data"""
        
        logger.info("\n" + "-"*70)
        logger.info("DATA PREPARATION")
        logger.info("-"*70)
        
        try:
            # Validate files exist
            validator = SafeDataValidator()
            for csv_path in [features_csv, hci_csv, incidence_csv]:
                if not validator.validate_csv_exists(csv_path):
                    raise FileNotFoundError(f"Required file missing: {csv_path}")
            
            # Load data
            logger.info("[DATA] Loading features...")
            features_df = pd.read_csv(features_csv)
            
            logger.info("[DATA] Loading HCI labels...")
            hci_labels_df = pd.read_csv(hci_csv)
            
            logger.info("[DATA] Loading incidence...")
            incidence_df = pd.read_csv(incidence_csv)
            
            # Extract hyperedge features
            hyperedge_features = features_df.drop('hyperedge_id', axis=1).values.astype(np.float32)
            
            # Build node features from hyperedges
            node_features = np.zeros((self.num_nodes, self.in_channels), dtype=np.float32)
            hyperedge_map = {
                he_id: i for i, he_id in enumerate(features_df['hyperedge_id'].values)
            }
            
            node_counts = np.zeros(self.num_nodes)
            for _, row in incidence_df.iterrows():
                he_idx = hyperedge_map.get(row['hyperedge_id'])
                node_idx = int(row['node_id'].split('_')[1]) % self.num_nodes
                
                if he_idx is not None and 0 <= node_idx < self.num_nodes:
                    node_features[node_idx] += hyperedge_features[he_idx]
                    node_counts[node_idx] += 1
            
            # Normalize by connection count
            for i in range(self.num_nodes):
                if node_counts[i] > 0:
                    node_features[i] /= node_counts[i]
            
            # Fill unconnected nodes
            unconnected_mask = node_counts == 0
            if unconnected_mask.any():
                avg_features = hyperedge_features.mean(axis=0)
                node_features[unconnected_mask] = avg_features
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(node_features).to(self.device)
            
            # Validate
            if not validator.validate_tensor(X_tensor, "Node features", (self.num_nodes, self.in_channels)):
                raise ValueError("Node features validation failed")
            
            # Normalize
            X_mean = X_tensor.mean(dim=0)
            X_std = X_tensor.std(dim=0) + 1e-8
            X_tensor = (X_tensor - X_mean) / X_std
            
            logger.info(f"[DATA] Node features: {X_tensor.shape}")
            
            # Build incidence matrix
            incidence_matrix = np.zeros((self.num_hyperedges, self.num_nodes), dtype=np.float32)
            for _, row in incidence_df.iterrows():
                he_idx = hyperedge_map.get(row['hyperedge_id'])
                node_idx = int(row['node_id'].split('_')[1]) % self.num_nodes
                if he_idx is not None and 0 <= node_idx < self.num_nodes:
                    incidence_matrix[he_idx, node_idx] = 1
            
            incidence_tensor = torch.FloatTensor(incidence_matrix).to(self.device)
            sparsity = (incidence_tensor.sum() / incidence_tensor.numel() * 100)
            logger.info(f"[DATA] Incidence matrix: {incidence_tensor.shape}, sparsity: {sparsity:.1f}%")
            
            # Edge index
            edges_i, edges_j = torch.nonzero(incidence_tensor, as_tuple=True)
            edge_index = torch.stack([edges_i, edges_j]).to(self.device)
            edge_types = ['supplies', 'uses', 'prices']
            assigned_edge_types = [
                edge_types[i % len(edge_types)] for i in range(edge_index.size(1))
            ]
            logger.info(f"[DATA] Edge index: {edge_index.shape}")
            
            # Node types
            node_types = []
            nodes_per_type = self.num_nodes // 3
            for i in range(self.num_nodes):
                if i < nodes_per_type:
                    node_types.append('supplier')
                elif i < 2 * nodes_per_type:
                    node_types.append('part')
                else:
                    node_types.append('transaction')
            logger.info(f"[DATA] Node types: {len(set(node_types))} types")
            
            # Target labels with safety
            y_price = np.random.normal(100, 20, self.num_nodes).astype(np.float32)
            y_change = np.random.uniform(-0.1, 0.1, self.num_nodes).astype(np.float32)
            
            hci_values = hci_labels_df['HCI'].values / hci_labels_df['HCI'].max()
            y_criticality = np.zeros(self.num_nodes, dtype=np.float32)
            y_criticality[:len(hci_values)] = hci_values
            
            y_price_tensor = torch.FloatTensor(y_price).to(self.device)
            y_change_tensor = torch.FloatTensor(y_change).to(self.device)
            y_criticality_tensor = torch.FloatTensor(y_criticality).to(self.device)
            
            # Validate labels
            for name, tensor in [
                ("Price", y_price_tensor),
                ("Change", y_change_tensor),
                ("Criticality", y_criticality_tensor)
            ]:
                if not validator.validate_tensor(tensor, f"{name} labels"):
                    raise ValueError(f"{name} labels validation failed")
            
            logger.info(f"[DATA] Target labels validated")
            
            # Timestamps
            timestamps = torch.linspace(0, 10, self.num_nodes).to(self.device)
            
            logger.info(f"[DATA] Data preparation complete")
            
            return {
                'X': X_tensor,
                'incidence_matrix': incidence_tensor,
                'node_types': node_types,
                'edge_index': edge_index,
                'edge_types': assigned_edge_types,
                'timestamps': timestamps,
                'y_price': y_price_tensor,
                'y_change': y_change_tensor,
                'y_criticality': y_criticality_tensor
            }
        
        except Exception as e:
            logger.error(f"[ERROR] Data preparation failed: {e}")
            raise

    def prepare_data_with_moo(
        self,
        moo_embedding: np.ndarray,
        moo_knee_weights: np.ndarray,
        hic_targets: np.ndarray,
        features_csv: str = 'outputs/datasets/features.csv',
        hci_csv: str = 'outputs/datasets/hci_labels.csv',
        incidence_csv: str = 'outputs/datasets/incidence.csv'
    ) -> Dict:
        """
        Prepare data with MOO transfer mechanisms integrated

        Args:
            moo_embedding: (4,) MOO knee-point embedding to append to features
            moo_knee_weights: (4,) MOO solution for loss weight initialization
            hic_targets: (N,) HIC cascade targets for KL divergence

        Returns:
            Dict with MOO-integrated data including:
            - node_features: Enhanced with MOO features if enabled
            - loss_weights: MOO-calibrated weights if enabled
            - cascade_targets: HIC targets if enabled
        """

        logger.info("\n" + "-"*70)
        logger.info(f"MOO DATA PREPARATION - {self.moo_config.name}")
        logger.info("-"*70)

        # Start with standard data preparation
        data = self.prepare_data(features_csv, hci_csv, incidence_csv)

        # --- MOO MECHANISM 1: Feature-level integration ---
        if self.moo_config.use_moo_features:
            logger.info("[MOO] Integrating 4-dim MOO embedding into node features")

            # Get current node features
            node_features = data['X'].cpu().numpy()  # (N, 18)

            # Expand MOO embedding to all nodes
            moo_features = np.tile(moo_embedding, (self.num_nodes, 1))  # (N, 4)

            # Concatenate: (N, 18) + (N, 4) = (N, 22)
            enhanced_features = np.concatenate([node_features, moo_features], axis=1)

            # Update tensor
            data['X'] = torch.FloatTensor(enhanced_features).to(self.device)

            logger.info(f"   Original features: {node_features.shape}")
            logger.info(f"   MOO embedding: {moo_embedding.shape}")
            logger.info(f"   Enhanced features: {enhanced_features.shape}")
        else:
            logger.info("[MOO] Feature integration DISABLED")

        # --- MOO MECHANISM 2: Loss weight initialization ---
        if self.moo_config.use_moo_loss_weights:
            logger.info("[MOO] Using MOO knee-point for loss weight initialization")

            # Update loss function weights
            self.loss_fn.weight_price = float(moo_knee_weights[0])
            self.loss_fn.weight_change = float(moo_knee_weights[1])
            self.loss_fn.weight_criticality = float(moo_knee_weights[2])
            # Note: 4th weight could be for a future task

            loss_weights = moo_knee_weights[:3]

            logger.info(f"   Price weight: {loss_weights[0]:.4f}")
            logger.info(f"   Change weight: {loss_weights[1]:.4f}")
            logger.info(f"   Criticality weight: {loss_weights[2]:.4f}")
        else:
            logger.info("[MOO] Loss weight integration DISABLED - using defaults")
            loss_weights = np.array([1.0, 0.5, 0.3])

        # Add to data dict
        data['loss_weights'] = torch.FloatTensor(loss_weights).to(self.device)

        # --- MOO MECHANISM 3: HIC cascade targets ---
        if self.moo_config.use_hic_targets:
            logger.info("[MOO] Using HIC cascade simulation targets")

            # Ensure correct shape and convert to tensor
            if len(hic_targets) != self.num_nodes:
                logger.warning(f"   HIC targets shape mismatch: {len(hic_targets)} vs {self.num_nodes}")
                # Pad or truncate as needed
                if len(hic_targets) > self.num_nodes:
                    hic_targets = hic_targets[:self.num_nodes]
                else:
                    padding = np.zeros(self.num_nodes - len(hic_targets))
                    hic_targets = np.concatenate([hic_targets, padding])

            cascade_targets = torch.FloatTensor(hic_targets).to(self.device)

            logger.info(f"   HIC targets shape: {cascade_targets.shape}")
            logger.info(f"   HIC range: [{cascade_targets.min():.4f}, {cascade_targets.max():.4f}]")
        else:
            logger.info("[MOO] HIC target integration DISABLED - using zeros")
            cascade_targets = torch.zeros(self.num_nodes).to(self.device)

        # Add to data dict
        data['cascade_targets'] = cascade_targets

        # --- Final validation ---
        logger.info(f"\n[MOO] Final data validation:")
        logger.info(f"   Node features: {data['X'].shape}")
        logger.info(f"   Loss weights: {data['loss_weights'].shape}")
        logger.info(f"   Cascade targets: {data['cascade_targets'].shape}")
        logger.info(f"   Expected input channels: {self.moo_config.input_dim}")

        # Update model input channels if needed
        actual_channels = data['X'].shape[1]
        expected_channels = self.moo_config.input_dim

        if actual_channels != expected_channels:
            logger.warning(f"[MOO] Channel mismatch: {actual_channels} vs {expected_channels}")
            logger.warning("   Model may need re-initialization with correct input_dim")

        logger.info(f"[MOO] Data preparation complete with {self.moo_config.name} configuration")

        return data

    def train_epoch(self, data: Dict) -> Dict:
        """Safely train single epoch with AMP"""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            if self.use_amp:
                with autocast(device_type='cuda'):
                    output = self.model(
                        node_features=data['X'],
                        incidence_matrix=data['incidence_matrix'],
                        node_types=data['node_types'],
                        edge_index=data['edge_index'],
                        edge_types=data['edge_types'],
                        timestamps=data['timestamps']
                    )
                    
                    loss_dict = self.loss_fn(
                        price_pred=output['price_pred'],
                        price_target=data['y_price'],
                        change_pred=output['change_pred'],
                        change_target=data['y_change'],
                        criticality_pred=output['criticality'],
                        criticality_target=data['y_criticality']
                    )

                    # MOO Mechanism 3: Add HIC cascade KL divergence loss
                    if self.moo_config.use_hic_targets and 'cascade_targets' in data:
                        cascade_pred = torch.softmax(output['cascade_scores'], dim=0)
                        cascade_target = torch.softmax(data['cascade_targets'], dim=0)

                        kl_loss = nn.functional.kl_div(
                            cascade_pred.log(), cascade_target, reduction='batchmean'
                        )

                        # Add to total loss with small weight
                        loss_dict['cascade_kl'] = kl_loss
                        loss_dict['total_loss'] = loss_dict['total_loss'] + 0.1 * kl_loss
                
                self.scaler.scale(loss_dict['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    node_features=data['X'],
                    incidence_matrix=data['incidence_matrix'],
                    node_types=data['node_types'],
                    edge_index=data['edge_index'],
                    edge_types=data['edge_types'],
                    timestamps=data['timestamps']
                )
                
                loss_dict = self.loss_fn(
                    price_pred=output['price_pred'],
                    price_target=data['y_price'],
                    change_pred=output['change_pred'],
                    change_target=data['y_change'],
                    criticality_pred=output['criticality'],
                    criticality_target=data['y_criticality']
                )

                # MOO Mechanism 3: Add HIC cascade KL divergence loss
                if self.moo_config.use_hic_targets and 'cascade_targets' in data:
                    cascade_pred = torch.softmax(output['cascade_scores'], dim=0)
                    cascade_target = torch.softmax(data['cascade_targets'], dim=0)

                    kl_loss = nn.functional.kl_div(
                        cascade_pred.log(), cascade_target, reduction='batchmean'
                    )

                    # Add to total loss with small weight
                    loss_dict['cascade_kl'] = kl_loss
                    loss_dict['total_loss'] = loss_dict['total_loss'] + 0.1 * kl_loss
                
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            return loss_dict
        
        except RuntimeError as e:
            logger.error(f"[ERROR] Training error: {e}")
            SafeDeviceManager.clear_cache()
            raise
    
    def train(
        self,
        data: Dict,
        epochs: int = 50,
        verbose: bool = True,
        save_interval: int = 10
    ) -> Dict:
        """Safe training loop"""
        
        logger.info("\n" + "-"*70)
        logger.info("TRAINING")
        logger.info("-"*70)
        
        best_loss = float('inf')
        
        try:
            for epoch in range(self.start_epoch, epochs):
                try:
                    loss_dict = self.train_epoch(data)
                    
                    # Update history
                    self.history['loss'].append(loss_dict['total_loss'].item())
                    self.history['loss_price'].append(loss_dict['loss_price'])
                    self.history['loss_change'].append(loss_dict['loss_change'])
                    self.history['loss_criticality'].append(loss_dict['loss_criticality'])
                    self.history['cascade_kl'].append(loss_dict.get('cascade_kl', 0.0))
                    self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                    
                    # Scheduler step
                    self.scheduler.step()
                    
                    # Verbose output
                    if verbose and (epoch + 1) % 10 == 0:
                        current_loss = loss_dict['total_loss'].item()
                        is_best = current_loss < best_loss
                        if is_best:
                            best_loss = current_loss
                        
                        logger.info(f"\n[EPOCH] {epoch+1}/{epochs}")
                        logger.info(f"  Total Loss:       {current_loss:.6f} {'(BEST)' if is_best else ''}")
                        logger.info(f"  Price Loss:       {loss_dict['loss_price']:.6f}")
                        logger.info(f"  Change Loss:      {loss_dict['loss_change']:.6f}")
                        logger.info(f"  Criticality Loss: {loss_dict['loss_criticality']:.6f}")
                        if 'cascade_kl' in loss_dict:
                            logger.info(f"  Cascade KL Loss:  {loss_dict['cascade_kl']:.6f}")
                        logger.info(f"  LR:               {self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    # Save checkpoint
                    if (epoch + 1) % save_interval == 0:
                        is_best = loss_dict['total_loss'].item() < best_loss
                        self.save_checkpoint(epoch + 1, is_best=is_best)
                
                except Exception as e:
                    logger.error(f"[ERROR] Epoch {epoch+1} failed: {e}")
                    SafeDeviceManager.clear_cache()
                    raise
            
            logger.info(f"\n[TRAINING] Completed")
            logger.info(f"  Final loss: {self.history['loss'][-1]:.6f}")
            logger.info(f"  Best loss: {best_loss:.6f}")
            
            return self.history
        
        except KeyboardInterrupt:
            logger.warning("\n[INTERRUPT] Training interrupted by user")
            self.save_checkpoint(epoch, is_best=True)
            raise
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            self.save_checkpoint(epoch) if 'epoch' in locals() else None
            raise


def main():
    """Safe main execution with MOO ablation support"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HT-HGNN Training with MOO Ablation Study')
    parser.add_argument(
        '--moo_mode',
        type=str,
        default='full',
        choices=['full', 'no_feature', 'no_loss', 'no_hic', 'none', 'ablation_study'],
        help='MOO transfer mechanism mode (default: full). Use "ablation_study" to run all configurations.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--ablation_epochs',
        type=int,
        default=30,
        help='Number of epochs per config in ablation study (default: 30)'
    )
    parser.add_argument(
        '--save_models',
        action='store_true',
        help='Save models during ablation study'
    )
    parser.add_argument(
        '--ssl_temperature',
        type=float,
        default=0.1,
        help='Temperature parameter (tau) for NT-Xent contrastive loss (default: 0.1)'
    )
    parser.add_argument(
        '--attention_type',
        type=str,
        default='structural',
        choices=['uniform', 'scalar', 'structural'],
        help='Hyperedge aggregation method: uniform (mean pooling), scalar (learned attention), structural (distance-based) (default: structural)'
    )

    args = parser.parse_args()

    # Initialize MOO configuration
    if args.moo_mode == 'ablation_study':
        moo_config = None  # Will run all configs
    else:
        moo_config = CONFIG_MAP.get(args.moo_mode, CONFIG_FULL)

    start_time = datetime.now()
    logger.info(f"🚀 Session started: {start_time}")

    if args.moo_mode == 'ablation_study':
        logger.info(f"🔬 Mode: Full MOO Ablation Study ({len(CONFIG_MAP)} configurations)")
        logger.info(f"⚙️  Epochs per config: {args.ablation_epochs}")
    else:
        logger.info(f"🎯 Mode: Single Configuration - {moo_config.name}")
        logger.info(f"🗲 Mechanisms: {', '.join(moo_config.enabled_mechanisms) if moo_config.enabled_mechanisms else 'None (Pure Neural)'}")
        logger.info(f"⚙️  Epochs: {args.epochs}")

    logger.info(f"🌡️ SSL Temperature (τ): {args.ssl_temperature}")
    logger.info(f"🔗 Attention Type: {args.attention_type}")

    try:
        if args.moo_mode == 'ablation_study':
            # Run full ablation study
            results = run_moo_ablation_study_pipeline(
                epochs=args.ablation_epochs,
                save_models=args.save_models,
                ssl_temperature=args.ssl_temperature,
                attention_type=args.attention_type
            )
            logger.info(f"🎉 Ablation study completed with {len(results)} results")
        else:
            # Run single configuration training
            results = train_single_moo_config(
                moo_config=moo_config,
                epochs=args.epochs,
                ssl_temperature=args.ssl_temperature,
                attention_type=args.attention_type
            )
            logger.info(f"🎉 Single configuration training completed")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Mode: {args.moo_mode}")

        if args.moo_mode != 'ablation_study':
            logger.info(f"Final Loss: {results.get('final_loss', 'N/A')}")
            logger.info(f"Best Validation Accuracy: {results.get('best_val_acc', 'N/A')}")
        else:
            logger.info(f"Configurations tested: {len(results)}")

        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n[FATAL] Training failed: {e}")
        logger.error("[FATAL] Check logs for details.")
        raise


def train_single_moo_config(
    moo_config: MOOTransferConfig,
    epochs: int = 50,
    ssl_temperature: float = 0.1,
    attention_type: str = 'structural'
) -> Dict:
    """
    Train HT-HGNN with a specific MOO configuration.

    Args:
        moo_config: MOO transfer configuration
        epochs: Number of training epochs
        ssl_temperature: Temperature parameter for NT-Xent loss
        attention_type: Hyperedge aggregation method

    Returns:
        Dictionary with training results
    """
    logger.info(f"🔧 Initializing trainer for {moo_config.name}")

    # Generate synthetic MOO data (replace with real MOO outputs)
    moo_embedding = torch.randn(1206, 4)  # 4-dim MOO knee-point embedding
    moo_knee_weights = torch.tensor([1.2, 0.9, 1.1, 0.7])  # MOO-optimized loss weights
    hic_targets = torch.randn(1206, 4)  # MOO-calibrated HIC cascade targets
    hic_targets = torch.softmax(hic_targets, dim=1)  # Normalize to probabilities

    # Initialize trainer
    trainer = SafeHT_HGNN_Trainer(
        in_channels=moo_config.input_dim,  # Always 16 (12 raw + 4 MOO or zero-padded)
        hidden_channels=64,
        out_channels=32,
        num_nodes=1206,
        num_hyperedges=36,
        learning_rate=0.001,
        weight_decay=1e-5,
        use_amp=True,
        moo_config=moo_config,
        ssl_temperature=ssl_temperature,
        attention_type=attention_type
    )

    # Prepare data with MOO integration
    data = trainer.prepare_data_with_moo(
        moo_embedding=moo_embedding,
        moo_knee_weights=moo_knee_weights,
        hic_targets=hic_targets
    )

    logger.info(f"📊 Training with {moo_config.name} configuration")
    logger.info(f"   Feature dims: {data['node_features'].shape[1]}")
    logger.info(f"   Loss weights: {data['loss_weights'].tolist()}")
    logger.info(f"   HIC targets shape: {data['cascade_targets'].shape}")

    # Train
    history = trainer.train(data, epochs=epochs, verbose=True, save_interval=10)

    # Save final model
    trainer.save_checkpoint(epochs, is_best=True)

    # Save history with MOO config info
    history_path = Path(f'outputs/training_history_{moo_config.name.lower().replace(" ", "_")}.json')
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history_to_save = {
        'config_name': moo_config.name,
        'config_description': moo_config.description,
        'enabled_mechanisms': moo_config.enabled_mechanisms,
        'history': {
            k: [float(v) if isinstance(v, np.floating) else v for v in vals]
            for k, vals in history.items()
        }
    }

    with open(history_path, 'w') as f:
        json.dump(history_to_save, f, indent=2)

    logger.info(f"📁 Training history saved to {history_path}")

    return {
        'final_loss': history['loss'][-1],
        'best_val_acc': max(history.get('val_accuracy', [0.0])) if 'val_accuracy' in history else 0.0,
        'convergence_epoch': len(history['loss']),
        'config_name': moo_config.name
    }


def run_moo_ablation_study_pipeline(
    epochs: int = 30,
    save_models: bool = False,
    ssl_temperature: float = 0.1,
    attention_type: str = 'structural'
) -> pd.DataFrame:
    """
    Run complete MOO ablation study pipeline.

    Args:
        epochs: Number of epochs per configuration
        save_models: Whether to save trained models
        ssl_temperature: Temperature parameter for NT-Xent loss
        attention_type: Hyperedge aggregation method

    Returns:
        DataFrame with ablation study results
    """
    logger.info("🔬 Starting MOO Ablation Study")
    logger.info("="*60)

    # Prepare synthetic MOO data (replace with real MOO outputs)
    moo_embedding = torch.randn(1206, 4)
    moo_knee_weights = torch.tensor([1.2, 0.9, 1.1, 0.7])
    hic_targets = torch.randn(1206, 4)
    hic_targets = torch.softmax(hic_targets, dim=1)

    # Prepare synthetic data (replace with real data loading)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = {
        'features': torch.randn(1000, 12),  # Raw features before MOO enhancement
        'labels': torch.randint(0, 4, (1000,)),
        'incidence_matrix': torch.randn(36, 1000),
        'node_types': ['supplier'] * 334 + ['part'] * 333 + ['transaction'] * 333,
        'edge_index': torch.randint(0, 1000, (2, 2000)),
        'edge_types': ['supplies', 'uses', 'prices'] * 667,
        'timestamps': torch.linspace(0, 10, 1000),
        'num_hyperedges': 36
    }

    val_data = {
        'features': torch.randn(200, 12),
        'labels': torch.randint(0, 4, (200,)),
        'incidence_matrix': torch.randn(36, 200),
        'node_types': ['supplier'] * 67 + ['part'] * 67 + ['transaction'] * 66,
        'edge_index': torch.randint(0, 200, (2, 400)),
        'edge_types': ['supplies', 'uses', 'prices'] * 134,
        'timestamps': torch.linspace(0, 10, 200),
        'num_hyperedges': 36
    }

    test_data = {
        'features': torch.randn(100, 12),
        'labels': torch.randint(0, 4, (100,)),
        'incidence_matrix': torch.randn(36, 100),
        'node_types': ['supplier'] * 34 + ['part'] * 33 + ['transaction'] * 33,
        'edge_index': torch.randint(0, 100, (2, 200)),
        'edge_types': ['supplies', 'uses', 'prices'] * 67,
        'timestamps': torch.linspace(0, 10, 100),
        'num_hyperedges': 36
    }

    # Run ablation study
    results = run_moo_ablation_study(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        base_model_class=HeterogeneousTemporalHypergraphNN,
        moo_embedding=moo_embedding[:train_data['features'].size(0)],  # Match train size
        moo_knee_weights=moo_knee_weights,
        hic_targets=hic_targets[:train_data['features'].size(0)],  # Match train size
        epochs=epochs,
        device=device,
        save_models=save_models,
        output_dir='outputs/moo_ablation'
    )

    return results


if __name__ == "__main__":
    main()
