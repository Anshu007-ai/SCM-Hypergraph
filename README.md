<div align="center">

# Supply Chain Risk Analysis using HT-HGNN v2.0

**Heterogeneous Temporal Hypergraph Neural Networks for Supply Chain Disruption Prediction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Department of Mechanical Engineering — Final Year Project 2025–2026*

**Team: Anshu & Param**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Quick Start (Docker)](#quick-start-docker)
- [Manual Setup](#manual-setup)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [How It Works — Prediction Pipeline](#how-it-works--prediction-pipeline)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Training](#training)
- [Evaluation & Results](#evaluation--results)
- [Sample Data for Testing](#sample-data-for-testing)
- [Scripts](#scripts)
- [License](#license)

---

## Overview

**HT-HGNN v2.0** is a production-ready machine learning platform that predicts supply chain disruptions *before* they happen. Unlike traditional pairwise graph neural networks, it models complex **multi-party supply chain interactions** using **hypergraph neural networks** — where a single hyperedge can connect an entire supplier cluster, production line, or shipping corridor simultaneously.

The system combines:
- **Spectral Hypergraph Convolution** — multi-way relationship learning (Zhou et al. formulation)
- **Temporal Fusion Encoder** — Bi-LSTM + Transformer with learned gating for temporal dynamics
- **Heterogeneous Relation Attention** — 5 distinct relation types with softmax attention
- **Cascade Disruption Simulation** — Hypergraph Independent Cascade model
- **HyperSHAP Explainability** — Custom SHAP for hypergraphs (node, hyperedge, feature attribution)

---

## Key Innovation

| Aspect | Traditional GNN | **HT-HGNN v2.0** |
|--------|----------------|-------------------|
| Relationships | Pairwise edges only | **Hyperedges** connecting multiple nodes |
| Temporal | Static snapshots | **Bi-LSTM + Transformer** fusion |
| Relations | Homogeneous | **5 heterogeneous types** with attention |
| Prediction | Single-task | **4 simultaneous outputs** (price, disruption, criticality, cascade) |
| Explainability | Post-hoc only | **Built-in HyperSHAP** |
| Simulation | None | **Live cascade disruption engine** |

---

## Features

- **Interactive Web Dashboard** — Upload CSV data and see predictions in real-time charts (criticality distribution, price predictions, scatter plots, top at-risk nodes)
- **Multi-Dataset Support** — Trained and validated on 6 diverse real-world datasets
- **Live Cascade Simulation** — Simulate disruption propagation with configurable shock parameters and animated node-link diagrams
- **MOO Pareto Analysis** — Interactive 4-objective Pareto front visualisation with weight sliders for ATC decision-making trade-offs
- **IndiGo Case Study Timeline** — Animated 7-stage disruption timeline for the Nov 2025 IndiGo crisis with OTP degradation charts
- **HyperSHAP Explainability** — Understand *why* each node is flagged as risky
- **REST API** — 14 endpoints including WebSocket for live training stream
- **Docker Deployment** — One-command deployment with `docker compose up`
- **Async Task Queue** — Celery + Redis for long-running simulations

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18 + Vite)                │
│              Tailwind CSS + Framer Motion + Recharts         │
│          Served by nginx:alpine on port 5173                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP / WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│                  Backend (FastAPI + Uvicorn)                  │
│                    Port 8000 — 14 endpoints                   │
│  ┌──────────┬──────────┬───────────┬──────────┬───────────┐  │
│  │ /predict │ /explain │ /simulate │/datasets │ /upload/  │  │
│  │          │          │ /cascade  │  /{id}   │  predict  │  │
│  └──────────┴──────────┴───────────┴──────────┴───────────┘  │
└──────┬──────────────────┬───────────────────────────────────┘
       │                  │
┌──────▼──────┐   ┌───────▼────────┐
│  Redis 7    │   │ Celery Worker  │
│  Cache &    │   │ Async tasks:   │
│  Broker     │   │ simulation,    │
│  Port 6379  │   │ stress test    │
└─────────────┘   └────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch 2.1+, PyTorch Geometric |
| **Model** | Spectral HypergraphConv, Bi-LSTM, Transformer, HGT |
| **Backend** | FastAPI 0.104+, Uvicorn, Pydantic v2 |
| **Frontend** | React 18, TypeScript 5.2, Vite, Tailwind CSS 3.3, Framer Motion 11, Recharts |
| **Task Queue** | Celery 5.3, Redis 7 |
| **Containerization** | Docker, Docker Compose |
| **Baselines** | XGBoost, Random Forest, Gradient Boosting, scikit-learn |
| **Explainability** | Custom HyperSHAP, Integrated Gradients, SHAP |

---

## Quick Start (Docker)

> **Prerequisites:** [Docker Desktop](https://docker.com/products/docker-desktop) installed and running.

```bash
# Clone the repository
git clone <repo-url>
cd SChypergraph

# Build and start all 4 services
docker compose up --build

# Access:
#   Frontend  → http://localhost:5173
#   Backend   → http://localhost:8000
#   API Docs  → http://localhost:8000/docs
```

To stop:

```bash
docker compose down
```

---

## Manual Setup

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# For CPU-only PyTorch (no CUDA required):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Start backend server
python frontend/backend_server.py
# → Running on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → Running on http://localhost:5173
```

### Celery Worker (optional, for async simulations)

```bash
celery -A src.tasks.celery_app worker --loglevel=info
```

---

## Project Structure

```
SChypergraph/
├── src/                          # Python source modules
│   ├── data/                     # Dataset loaders & adapters
│   │   ├── data_adapter.py       #   Unified normalization layer
│   │   ├── dataco_loader.py      #   DataCo logistics (180K records)
│   │   ├── bom_loader.py         #   Automotive BOM (12K components)
│   │   ├── indigo_disruption_loader.py #   IndiGo aviation disruption (84 nodes)
│   │   ├── port_loader.py        #   Global port disruptions (847 ports)
│   │   ├── maintenance_loader.py #   Predictive maintenance (10K records)
│   │   └── retail_loader.py      #   M5 retail demand-supply (30K products)
│   ├── models/                   # Neural network architectures
│   │   ├── ht_hgnn_model.py      #   v1.0 full model
│   │   ├── hypergraph_conv.py    #   v2.0 SpectralHypergraphConv
│   │   ├── temporal_encoder.py   #   v2.0 TemporalFusionEncoder
│   │   ├── relation_fusion.py    #   v2.0 HeterogeneousRelationFusion
│   │   ├── cascade_model.py      #   v2.0 CascadeRiskHead
│   │   └── baseline_models.py    #   XGBoost, RF, GBM baselines
│   ├── hypergraph/               # Hypergraph data structures
│   │   ├── hypergraph.py         #   Core Hypergraph class
│   │   ├── dynamic_constructor.py#   Dynamic hyperedge mining
│   │   ├── risk_labels.py        #   HCI risk label generation
│   │   └── visualization_utils.py#   Graph serialization (D3/Cytoscape)
│   ├── explainability/           # Model interpretability
│   │   ├── hypershap.py          #   HyperSHAP for hypergraphs
│   │   ├── hyperedge_importance.py#  Gradient/removal/attention methods
│   │   └── feature_attribution.py#   Integrated gradients
│   ├── simulation/               # Disruption simulation
│   │   ├── cascade_engine.py     #   Hypergraph Independent Cascade
│   │   ├── stress_tester.py      #   Bulk scenario testing
│   │   └── scenario_builder.py   #   What-if scenario construction
│   ├── evaluation/               # Metrics & validation
│   │   ├── validation.py         #   Ablation & failure simulation
│   │   ├── hypergraph_metrics.py #   Hyperedge-aware accuracy, NDCG
│   │   └── cascade_eval.py       #   Cascade depth MAE, spread accuracy
│   └── tasks/                    # Celery async tasks
│       ├── celery_app.py         #   Celery + Redis config
│       └── simulation_tasks.py   #   Async cascade/stress tasks
├── frontend/                     # Web application
│   ├── backend_server.py         #   FastAPI server (14 endpoints)
│   ├── src/
│   │   ├── App.tsx               #   Main showcase + interactive demo
│   │   ├── main.tsx              #   React entry point
│   │   ├── index.css             #   Dark theme + Tailwind utilities
│   │   ├── components/           #   Reusable UI components
│   │   │   ├── ParetoVisualiser.tsx   # MOO Pareto front with weight sliders
│   │   │   ├── CascadeSimulation.tsx  # SVG node-link cascade animation
│   │   │   └── IndiGoTimeline.tsx     # Animated 7-stage disruption timeline
│   │   ├── pages/                #   Route pages
│   │   ├── hooks/                #   Custom React hooks
│   │   ├── services/             #   API service layer
│   │   └── types/                #   TypeScript type definitions
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── scripts/                      # CLI utilities
│   ├── download_datasets.py      #   Download all 6 datasets
│   ├── build_hypergraphs.py      #   Pre-build hypergraph structures
│   ├── simulate_cascade.py       #   Run cascade simulation
│   ├── stress_tester.py          #   Bulk stress testing
│   └── explain.py                #   Generate HyperSHAP explanations
├── tests/                        # Test suite (pytest)
│   ├── test_api_endpoints.py
│   ├── test_cascade_engine.py
│   ├── test_data_loaders.py
│   ├── test_hypergraph_conv.py
│   └── test_hypershap.py
├── outputs/                      # Generated outputs
│   ├── checkpoints/              #   Model checkpoints (best.pt, latest.pt)
│   ├── datasets/                 #   Processed data files
│   ├── models/                   #   Exported models
│   └── training_history.json     #   Training metrics
├── Data set/                     # Raw datasets
│   ├── DataCo/                   #   DataCoSupplyChainDataset.csv
│   ├── BOM/                      #   train_set.csv, test_set.csv
│   ├── IndiGo/                   #   indigo_disruption.csv
│   ├── BOM/                      #   train_set.csv, test_set.csv
│   └── Maintenance/              #   ai4i2020.csv
├── train_ht_hgnn.py              # Main training script (v2.0)
├── docker-compose.yml            # 4-service Docker Compose
├── Dockerfile.backend            # Python + PyTorch (CPU)
├── Dockerfile.frontend           # Node + Vite build → nginx
├── requirements.txt              # Python dependencies
└── sample_supply_chain_data.csv  # Sample test data (50 nodes)
```

---

## Datasets

The model is trained and validated on **6 diverse real-world supply chain datasets**:

| Dataset | Domain | Records | Nodes | Hyperedges | Time Span | Source |
|---------|--------|---------|-------|------------|-----------|--------|
| **DataCo Supply Chain** | E-commerce Logistics | 180,519 | 10,862 | 1,247 | 2015–2018 | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) |
| **Automotive BOM** | Manufacturing | 12,305 | 2,584 | 486 | 2020–2023 | [Kaggle](https://www.kaggle.com/datasets/willianoliveiragibin/tech-parts-orders) |
| **IndiGo Aviation Disruption** | Aviation Service Chain | 84 | 84 | 18 | Jan–Dec 2025 | [DGCA India](https://www.dgca.gov.in) + Synthetic |
| **Global Port Disruption** | Maritime Shipping | 847 | 456 | 312 | 2020–2024 | [Kaggle](https://www.kaggle.com/datasets/jeanmidev/world-ports) |
| **AI4I Maintenance** | Predictive Maintenance | 10,000 | 10,000 | 874 | Synthetic | [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) |
| **M5 Walmart Retail** | Retail Demand-Supply | 30,490 | 3,049 | 528 | 2011–2016 | [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy) |

---

## How It Works — Prediction Pipeline

```
CSV Data → DataAdapter → Hypergraph Construction → Spectral Conv → Temporal Fusion → Relation Attention → Risk Output
```

### Step-by-Step:

1. **Data Ingestion** — Raw supply chain data (orders, shipments, components) is loaded and normalized. Numeric features scaled to [0,1], categorical features encoded, timestamps aligned.

2. **Hypergraph Construction** — Unlike standard graphs with pairwise edges, hyperedges connect multiple related nodes simultaneously. A single hyperedge might connect a supplier + 3 manufacturers + 2 shipping routes that share risk. `DynamicHyperedgeConstructor` also mines temporal co-occurrence patterns.

3. **Spectral Hypergraph Convolution** — Node features propagate through the hypergraph using spectral convolution. Each node aggregates information from ALL members of its hyperedges — not just direct neighbors. Residual connections and LayerNorm stabilize deep propagation.

4. **Temporal Fusion** — Bi-LSTM captures sequential patterns (seasonal trends, delivery cycles). Transformer encoder captures long-range dependencies. A learned gating mechanism fuses both streams.

5. **Heterogeneous Relation Attention** — 5 relation types processed with type-specific attention weights: `supplier_of`, `manufactured_by`, `transported_by`, `quality_controlled_by`, `co_disrupted_with`.

6. **Multi-Task Risk Output** — Four prediction heads output simultaneously:
   - **Price Prediction** (MSE loss)
   - **Disruption Detection** — binary disruption signal (BCE loss)
   - **Criticality Classification** — Low / Medium / High / Critical (4-class CE loss)
   - **Cascade Risk Score** — spread probability (KL divergence loss)

**Loss weights:** `1.0×price + 0.8×change + 1.2×criticality + 0.6×cascade`

---

## API Reference

Base URL: `http://localhost:8000`

### Core Endpoints (v1.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model metadata & training metrics |
| `POST` | `/predict` | Run inference on node features |
| `POST` | `/analyze` | Full risk assessment with average risk |
| `POST` | `/upload/predict` | **Upload CSV file → get predictions** |
| `GET` | `/training/history` | Training loss history |

### v2.0 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/datasets` | List all 6 available datasets |
| `POST` | `/datasets/{id}/load` | Build hypergraph for a dataset |
| `POST` | `/explain` | HyperSHAP node attribution |
| `POST` | `/simulate/cascade` | Start async cascade simulation |
| `GET` | `/tasks/{task_id}` | Poll async task status |
| `WS` | `/training/stream` | WebSocket for live training metrics |

### Example: Upload CSV for Prediction

```bash
curl -X POST http://localhost:8000/upload/predict \
  -F "file=@sample_supply_chain_data.csv"
```

Response:
```json
{
  "pricePredictions": [523.4, 478.1, ...],
  "changePredictions": [0.03, -0.08, ...],
  "criticalityScores": [0.32, 0.78, ...],
  "nodeIds": ["node_0", "node_1", ...]
}
```

Interactive API docs available at: **http://localhost:8000/docs**

---

## Frontend

The frontend is a **professional showcase website + interactive demo** built with React 18, TypeScript, Tailwind CSS, Framer Motion, and Recharts.

### Key Sections

| Section | Description |
|---------|-------------|
| **Hero** | Animated network background, project title, CTA buttons |
| **Problem** | 3 cards explaining supply chain analysis challenges |
| **Solution** | Graph vs. Hypergraph visual comparison |
| **Architecture** | 5-step horizontal pipeline flow |
| **How It Predicts** | 6-step detailed prediction walkthrough |
| **IndiGo Crisis Simulation** | Interactive 3D hypergraph disruption simulation for aviation sector |
| **Datasets** | Expandable cards for all 6 datasets with source links |
| **Case Study: IndiGo Timeline** | Animated 7-stage Nov 2025 disruption timeline with OTP chart |
| **Try It Live** | **CSV upload → real-time charts** + interactive cascade simulation panel |
| **MOO Pareto Analysis** | 4-objective Pareto front with weight sliders and operating point selection |
| **Results** | Benchmark comparison + ablation study + detailed analysis |
| **Novelty** | 3 research contributions |
| **Team** | Team members |

### Live Demo Feature

1. Navigate to http://localhost:5173
2. Scroll to **"Try It Live"**
3. Upload any CSV with numeric columns (e.g., `sample_supply_chain_data.csv`)
4. Click **"Run Prediction"**
5. See interactive charts: criticality distribution, price predictions, price-vs-criticality scatter, top 10 at-risk nodes

---

## Training

```bash
# Train on a single dataset
python train_ht_hgnn.py --dataset dataco --epochs 100 --lr 0.001

# Transfer learning (pretrain on BOM, finetune on DataCo)
python train_ht_hgnn.py --dataset all --pretrain bom --finetune dataco

# Resume from checkpoint
python train_ht_hgnn.py --dataset dataco --resume outputs/checkpoints/best_dataco.pt
```

Checkpoints are saved to `outputs/checkpoints/`. Training history is logged to `outputs/training_history.json`.

---

## Evaluation & Results

### Benchmark Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 68.2% | 0.621 |
| XGBoost | 78.5% | 0.742 |
| GCN | 82.1% | 0.793 |
| GAT | 84.3% | 0.821 |
| T-GCN | 87.6% | 0.852 |
| HT-HGNN v1.0 | 91.2% | 0.879 |
| **HT-HGNN v2.0** | **94.7%** | **0.901** |

### Ablation Study

| Configuration | Accuracy | Δ from Full |
|---------------|----------|-------------|
| Full Model | 94.7% | — |
| − Spectral Conv | 85.7% | −9.0% |
| − Temporal Encoder | 88.3% | −6.4% |
| − Heterogeneous Relations | 90.1% | −4.6% |
| − Cascade Head | 91.5% | −3.2% |

### Key Metrics

- **Criticality Accuracy:** 94.7%
- **F1 Score:** 0.901 (macro-averaged)
- **Inference Time:** < 60ms per batch (CPU)
- **Model Size:** 218K parameters

---

## Sample Data for Testing

A ready-to-use test file is included: **`sample_supply_chain_data.csv`**

50 nodes representing a **real automotive manufacturing supply chain ecosystem** — from raw material suppliers through Tier 1–4 components, OEM assembly lines, to logistics and distribution.

### Node Categories

| Category | Nodes | Examples |
|----------|-------|---------|
| **Raw Materials (Tier 1)** | 6 | `Steel_Coil_POSCO`, `Aluminum_Ingot_Hindalco`, `Natural_Rubber_Thai`, `Lithium_Carbonate_SQM` |
| **Tier 2 Components** | 12 | `Ball_Bearing_SKF`, `Fuel_Injector_Bosch`, `Brake_Pad_Brembo`, `Alternator_Valeo` |
| **Tier 3 Sub-assemblies** | 10 | `Engine_Block_Nemak`, `Turbocharger_BorgWarner`, `ABS_Module_Bosch`, `Airbag_Inflator_Autoliv` |
| **Tier 4 Electronics** | 8 | `ECU_Microcontroller_NXP`, `LiDAR_Sensor_Velodyne`, `ADAS_Processor_Mobileye`, `BMS_Controller_TI` |
| **OEM Assembly** | 6 | `Engine_Assembly_Line`, `Body_Welding_Shop`, `Paint_Shop_PPG`, `EV_Battery_Pack_Assembly` |
| **Logistics & Distribution** | 8 | `Port_Chennai_Kamarajar`, `CKD_Export_Mumbai_JNPT`, `Dealer_Network_South`, `Last_Mile_Delhivery` |

### Feature Correlations (visible in the graphs)

The data is designed to show **realistic supply chain risk correlations**:

- **Tier 4 electronics** (semiconductors, sensors) → highest `unit_cost`, lowest `substitutability`, highest `geographic_concentration` (Taiwan/Japan fabs), highest `disruption_frequency`
- **Raw materials** (steel, rubber, polymers) → low cost, high `supplier_reliability`, high `substitutability`, low disruption risk
- **High `geographic_concentration` → high `disruption_frequency`** (single-source risk pattern)
- **Low `substitutability` → high `demand_volatility`** (no alternatives amplify demand shocks)
- **Higher `tier_level` → longer `lead_time_days`** and lower `safety_stock_days`
- **Cascade path visible**: `Lithium_Carbonate_SQM` → `BMS_Controller_TI` → `EV_Battery_Pack_Assembly` (lithium shortage cascades through the EV battery chain)

### Features

| Column | Description | Range |
|--------|-------------|-------|
| `unit_cost` | Component cost ($) | $18.50–$170 |
| `supplier_reliability` | Reliability score | 0.66–0.97 |
| `lead_time_days` | Delivery lead time | 2–38 days |
| `tier_level` | Supply chain tier | 1–4 |
| `substitutability` | Replacement ease | 0.08–0.96 |
| `demand_volatility` | Demand fluctuation | 0.08–0.92 |
| `quality_reject_rate` | Defect rate | 0.005–0.10 |
| `safety_stock_days` | Buffer inventory | 2–25 days |
| `geographic_concentration` | Sourcing concentration | 0.10–0.96 |
| `disruption_frequency` | Historical disruption rate | 0.02–0.45 |

Upload it in the **"Try It Live"** section of the frontend to test the system. The backend extracts real node names from the CSV and displays them in all charts and the risk table.

### IndiGo Aviation Disruption Dataset (Dataset 6)

The **IndiGo Aviation Disruption 2025** dataset (`Data set/IndiGo/indigo_disruption.csv`) models the December 2025 IndiGo scheduling crisis — a real-world multi-layer supply chain cascade in India's aviation sector.

**Background:** DGCA's FDTL Phase 2 rules (effective Nov 1, 2025) reduced maximum pilot flying hours while IndiGo had ~60–70 aircraft grounded due to Pratt & Whitney powder-metal engine contamination. The combined shock caused ~4,500 flight cancellations over 10 days, affecting 9.82 lakh passengers, with OTP dropping from 84% → 62.7%.

**Cascade path:** `FDTL Phase 2 Activation → Pilot Roster Buffer Collapse → Route Cancellations (Hub Airports) → Passenger Displacement (600K) → Railway Demand Surge → Competitor Fare Spike → Regulatory Intervention (10% Schedule Cut) → Market Share Redistribution`

Run the cascade simulation with the built-in preset:
```bash
python scripts/simulate_cascade.py --preset indigo-fdtl
```

---

## Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `scripts/download_datasets.py` | Download all 6 datasets | `python scripts/download_datasets.py` |
| `scripts/build_hypergraphs.py` | Pre-build hypergraph structures | `python scripts/build_hypergraphs.py` |
| `scripts/simulate_cascade.py` | Run cascade simulation | `python scripts/simulate_cascade.py --dataset bom --shock-node NODE_001` |
| `scripts/stress_tester.py` | Bulk stress testing | `python scripts/stress_tester.py --dataset dataco --scenarios 100` |
| `scripts/explain.py` | Generate HyperSHAP explanations | `python scripts/explain.py --dataset dataco --node-ids N001 N002` |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_hypergraph_conv.py -v
```

---

## License

This project is developed as a Final Year Engineering Project for the Department of Mechanical Engineering (2025–2026).

---

<div align="center">
<strong>Built by Anshu & Param</strong><br>
<sub>Supply Chain Risk Analysis using HT-HGNN v2.0</sub>
</div>
