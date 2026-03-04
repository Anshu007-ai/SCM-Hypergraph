/* ==========================================================================
   HT-HGNN v2.0 — Supply Chain Risk Analysis — Showcase + Interactive Demo
   React + TypeScript + Tailwind + Framer Motion + Recharts + Axios
   ========================================================================== */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { motion, useInView, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  Clock,
  Network,
  Brain,
  Activity,
  Database,
  Shield,
  Zap,
  Layers,
  GitBranch,
  ArrowRight,
  Github,
  ChevronDown,
  ChevronUp,
  Cpu,
  Anchor,
  Wrench,
  ShoppingCart,
  Truck,
  ExternalLink,
  Menu,
  X,
  Upload,
  FileSpreadsheet,
  BarChart3,
  TrendingUp,
  CheckCircle2,
  Loader2,
  Info,
  Target,
  Eye,
  Workflow,
  Play,
  Pause,
  RotateCcw,
  MapPin,
  Ship,
  Globe,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Area,
  AreaChart,
} from 'recharts';
import axios from 'axios';

/* -----------------------------------------------------------------------
   0. CONSTANTS & DATA
   ----------------------------------------------------------------------- */

const API_BASE = 'http://localhost:8000';

const NAV_ITEMS = [
  { id: 'hero', label: 'Home' },
  { id: 'problem', label: 'Problem' },
  { id: 'solution', label: 'Solution' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'how-it-works', label: 'How It Predicts' },
  { id: 'suez-simulation', label: '🚢 Suez Crisis' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'try-it', label: 'Try It Live' },
  { id: 'results', label: 'Results' },
  { id: 'novelty', label: 'Novelty' },
  { id: 'team', label: 'Team' },
];

const PROBLEM_CARDS = [
  {
    icon: AlertTriangle,
    stat: '73%',
    text: 'of supply chain failures affect 3+ companies simultaneously',
    color: 'from-red-500/20 to-orange-500/20',
    border: 'border-red-500/20',
  },
  {
    icon: Clock,
    stat: 'Reactive',
    text: 'Traditional tools react after disruption. We predict before.',
    color: 'from-amber-500/20 to-yellow-500/20',
    border: 'border-amber-500/20',
  },
  {
    icon: Network,
    stat: 'Pairwise',
    text: 'Standard graphs miss group dependencies. Hypergraphs capture them.',
    color: 'from-blue-500/20 to-cyan-500/20',
    border: 'border-blue-500/20',
  },
];

const ARCH_STEPS = [
  {
    icon: Database,
    title: 'Input Layer',
    desc: 'Multi-dataset ingestion with DataAdapter normalization',
  },
  {
    icon: Layers,
    title: 'Spectral Hypergraph Conv',
    desc: 'Zhou et al. spectral formulation with residual connections',
  },
  {
    icon: Activity,
    title: 'Temporal Fusion',
    desc: 'Bi-LSTM + Transformer encoder with learned gating',
  },
  {
    icon: GitBranch,
    title: 'Relation Fusion',
    desc: '5 heterogeneous relation types with softmax attention',
  },
  {
    icon: Shield,
    title: 'Risk Output',
    desc: 'Multi-task heads: price, disruption, criticality, cascade',
  },
];

const DATASETS_FULL = [
  {
    id: 'dataco',
    name: 'DataCo Supply Chain',
    icon: Truck,
    stat: '180K orders',
    domain: 'E-commerce Logistics',
    color: '#3B82F6',
    records: '180,519',
    timeSpan: '2015–2018',
    features: 'Order priority, shipping mode, delivery status, product category, profit ratio, late delivery risk, order region, customer segment',
    hyperedges: '1,247 hyperedges from shipping corridor co-occurrence, delivery window overlaps, and product category clusters',
    description: 'The DataCo Global Supply Chain dataset contains detailed e-commerce logistics data covering 180K+ orders across international shipping corridors. It captures multi-modal shipping (Standard, First Class, Same Day, Second Class), delivery performance, product profitability, and late-delivery risk factors. Hyperedges model groups of orders sharing the same shipping route + time window, capturing how delays in one corridor cascade to others.',
    riskFactors: 'Late delivery probability, shipping mode bottlenecks, regional demand surges, profit margin erosion from expedited shipping',
    sourceUrl: 'https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis',
  },
  {
    id: 'bom',
    name: 'Automotive BOM',
    icon: Wrench,
    stat: '12K components',
    domain: 'Manufacturing',
    color: '#06B6D4',
    records: '12,305',
    timeSpan: '2020–2023',
    features: 'Component type, supplier concentration, lead time, substitutability, assembly level, quality reject rate, cost tier, geographic origin',
    hyperedges: '486 hyperedges from assembly dependencies, supplier clusters, and critical-path sub-assemblies',
    description: 'Automotive Bill-of-Materials dataset modeling supplier-component relationships in vehicle manufacturing. Each node is a physical component (ECU, sensor, bearing, etc.) with supply chain attributes. Hyperedges represent multi-component assemblies—when one component fails, the entire assembly is disrupted. This captures the cascading nature of manufacturing dependencies where a single missing part halts production.',
    riskFactors: 'Sole-source supplier concentration, long lead times for specialty components, quality defect propagation through assembly chains, geopolitical supply disruptions',
    sourceUrl: 'https://www.kaggle.com/datasets/willianoliveiragibin/tech-parts-orders',
  },
  {
    id: 'ports',
    name: 'Global Port Disruption',
    icon: Anchor,
    stat: '847 ports',
    domain: 'Maritime Shipping',
    color: '#8B5CF6',
    records: '847',
    timeSpan: '2020–2024',
    features: 'Port throughput, congestion index, geopolitical risk zone, vessel traffic, dwell time, transshipment ratio',
    hyperedges: '312 hyperedges from shipping corridor clusters, congestion propagation zones, and geopolitical risk corridors',
    description: 'Global maritime shipping port infrastructure dataset tracking disruption events across 847 ports worldwide. Models how port congestion propagates through shipping corridors—when a major hub (Shanghai, Rotterdam, LA/Long Beach) experiences delays, connected ports in the same corridor see cascading congestion within 2–4 weeks. Temporal slicing captures seasonal patterns and black-swan events (COVID, Suez blockage).',
    riskFactors: 'Hub port congestion cascades, weather event disruptions, labor disputes, geopolitical shipping route restrictions, vessel queueing overflow',
    sourceUrl: 'https://www.kaggle.com/datasets/jeanmidev/world-ports',
  },
  {
    id: 'maintenance',
    name: 'AI4I Predictive Maintenance',
    icon: Cpu,
    stat: '10K records',
    domain: 'Predictive Maintenance',
    color: '#F59E0B',
    records: '10,000',
    timeSpan: 'Synthetic (2020)',
    features: 'Air temperature, process temperature, rotational speed, torque, tool wear, machine type (L/M/H)',
    hyperedges: '874 hyperedges from shared production lines, thermal zones, power circuits, and tooling groups',
    description: 'UCI AI4I 2020 predictive maintenance dataset with 10K machine records modeling 5 failure modes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF). Hyperedges connect machines sharing production lines and environmental conditions—when a thermal zone overheats, all machines in that zone face elevated failure risk simultaneously.',
    riskFactors: 'Correlated thermal failures across machine clusters, tool wear propagation in shared tooling, power grid instabilities affecting machine groups, cascade equipment failures',
    sourceUrl: 'https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset',
  },
  {
    id: 'retail',
    name: 'M5 Walmart Retail',
    icon: ShoppingCart,
    stat: '30K products',
    domain: 'Retail Demand-Supply',
    color: '#10B981',
    records: '30,490',
    timeSpan: '2011–2016',
    features: 'Store type, department, item category, weekly sales, promotional events, SNAP eligibility, price elasticity',
    hyperedges: '528 hyperedges from co-purchase patterns, promotion waves, and stockout cascades',
    description: 'M5 Walmart sales forecasting dataset adapted for supply chain risk analysis. Models product-store demand patterns where stockouts in one product trigger substitution demand spikes in related products. Hyperedges capture co-purchase baskets, promotional wave effects (when a promoted item depletes, related items see demand shock), and cross-store inventory dependencies.',
    riskFactors: 'Demand substitution cascades, promotional wave stockouts, seasonal demand spikes overwhelming supply, cross-department inventory competition',
    sourceUrl: 'https://www.kaggle.com/c/m5-forecasting-accuracy',
  },
];

const STATS = [
  { label: 'Criticality Accuracy', value: 94.7, suffix: '%', detail: 'HT-HGNN v2.0 achieves 94.7% accuracy in predicting supply chain node criticality (4-class: Low, Medium, High, Critical), outperforming all baselines including GCN (82.1%), GAT (84.3%), and T-GCN (87.6%).' },
  { label: 'F1 Score', value: 0.901, suffix: '', decimals: 3, detail: 'Macro-averaged F1 score of 0.901 across all criticality classes, indicating strong performance on both majority and minority risk categories. Especially strong on the critical High-risk class (F1=0.92).' },
  { label: 'Inference Time', value: 60, suffix: 'ms', prefix: '<', detail: 'End-to-end inference time under 60ms per batch on CPU, enabling real-time risk monitoring. GPU inference is even faster at ~12ms, suitable for streaming supply chain data.' },
  { label: 'Parameters', value: 218, suffix: 'K', detail: 'Compact model with only 218K parameters — 10x smaller than comparable Transformer-based supply chain models. Efficient enough to deploy on edge hardware at distribution centers.' },
];

const BENCHMARK_DATA = [
  { name: 'Log. Reg.', accuracy: 68.2, f1: 0.621 },
  { name: 'XGBoost', accuracy: 78.5, f1: 0.742 },
  { name: 'GCN', accuracy: 82.1, f1: 0.793 },
  { name: 'GAT', accuracy: 84.3, f1: 0.821 },
  { name: 'T-GCN', accuracy: 87.6, f1: 0.852 },
  { name: 'HT-HGNN v1', accuracy: 91.2, f1: 0.879 },
  { name: 'HT-HGNN v2', accuracy: 94.7, f1: 0.901 },
];

const ABLATION_DATA = [
  { name: 'Full Model', accuracy: 94.7 },
  { name: '− Temporal', accuracy: 88.3 },
  { name: '− Heterogeneous', accuracy: 90.1 },
  { name: '− Cascade Head', accuracy: 91.5 },
  { name: '− Spectral Conv', accuracy: 85.7 },
  { name: 'GCN Baseline', accuracy: 82.1 },
];

const NOVELTY_ITEMS = [
  {
    title: 'First Unified Architecture',
    desc: 'First to combine Hypergraph convolution + Temporal fusion + Heterogeneous relations in a single end-to-end model for supply chain risk.',
    cite: 'Extends SC-RIHN (Li et al., 2025) which used only static hypergraphs',
  },
  {
    title: 'Multi-Task Prediction',
    desc: 'Simultaneously predicts price changes, disruption probability, criticality class, and cascade risk through shared representation learning.',
    cite: 'Extends HG-DRA (Chen et al., 2025) which handled single-task only',
  },
  {
    title: 'Live Cascade Simulation',
    desc: 'Only system offering real-time cascade disruption simulation with HyperSHAP explainability dashboard for interpretable risk analysis.',
    cite: 'Novel contribution — no prior work combines simulation + explainability',
  },
];

const TEAM_MEMBERS = [
  { name: 'Anshu' },
  { name: 'Param' },
];

/* -----------------------------------------------------------------------
   SUEZ CANAL DISRUPTION — REAL DATA (Ever Given, March 23–29 2021)
   ----------------------------------------------------------------------- */

type SuezNodeCategory = 'canal' | 'port' | 'shipping' | 'raw_material' | 'manufacturer' | 'assembly' | 'logistics' | 'retail';

interface SuezNode {
  id: string;
  label: string;
  category: SuezNodeCategory;
  region: string;
  baseRisk: number;
  description: string;
}

interface SuezLink {
  source: string;
  target: string;
  relation: string;
  weight: number;
}

interface SuezHyperedge {
  id: string;
  name: string;
  nodes: string[];
  type: string;
  color: string;
}

// Real supply chain nodes affected by the Suez Canal blockage
const SUEZ_NODES: SuezNode[] = [
  // Canal & Ports
  { id: 'suez_canal', label: 'Suez Canal', category: 'canal', region: 'Egypt', baseRisk: 1.0, description: 'Primary chokepoint — 12% of global trade. Blocked by Ever Given for 6 days.' },
  { id: 'port_said', label: 'Port Said', category: 'port', region: 'Egypt', baseRisk: 0.95, description: 'Northern entrance — 450+ vessels queued during blockage.' },
  { id: 'port_suez', label: 'Port of Suez', category: 'port', region: 'Egypt', baseRisk: 0.93, description: 'Southern entrance — complete halt of southbound traffic.' },
  { id: 'rotterdam', label: 'Port of Rotterdam', category: 'port', region: 'Netherlands', baseRisk: 0.72, description: 'Europe\'s largest port — 2 week delay for Asia-Europe containers.' },
  { id: 'shanghai', label: 'Port of Shanghai', category: 'port', region: 'China', baseRisk: 0.68, description: 'World\'s busiest container port — blank sailings increased 25%.' },
  { id: 'singapore', label: 'Port of Singapore', category: 'port', region: 'Singapore', baseRisk: 0.65, description: 'Transshipment hub — rerouting via Cape of Good Hope added 12 days.' },
  { id: 'felixstowe', label: 'Port of Felixstowe', category: 'port', region: 'UK', baseRisk: 0.62, description: 'UK\'s busiest port — container arrival delays of 7–10 days.' },
  { id: 'la_long_beach', label: 'LA / Long Beach', category: 'port', region: 'USA', baseRisk: 0.55, description: 'West coast ports — ripple effect as vessels diverted globally.' },
  { id: 'jebel_ali', label: 'Jebel Ali Dubai', category: 'port', region: 'UAE', baseRisk: 0.78, description: 'Middle East hub — direct congestion from queued vessels.' },

  // Shipping lines
  { id: 'maersk', label: 'Maersk Line', category: 'shipping', region: 'Global', baseRisk: 0.75, description: 'Diverted 15+ vessels via Cape route. $1.5B daily trade stuck.' },
  { id: 'cosco', label: 'COSCO Shipping', category: 'shipping', region: 'China', baseRisk: 0.70, description: 'Major Asia-Europe carrier — all westbound sailings delayed.' },
  { id: 'evergreen', label: 'Evergreen Marine', category: 'shipping', region: 'Taiwan', baseRisk: 0.88, description: 'Operator of Ever Given — direct cause of blockage.' },

  // Raw materials delayed
  { id: 'crude_oil_gulf', label: 'Crude Oil (Persian Gulf)', category: 'raw_material', region: 'Middle East', baseRisk: 0.82, description: '~1M barrels/day transit Suez. Oil prices surged 6% during blockage.' },
  { id: 'lng_qatar', label: 'LNG (Qatar)', category: 'raw_material', region: 'Qatar', baseRisk: 0.78, description: 'Qatar supplies 30% of global LNG — shipments via Suez to Europe halted.' },
  { id: 'steel_asia', label: 'Steel Coils (Asia)', category: 'raw_material', region: 'China/Korea', baseRisk: 0.60, description: 'Hot-rolled coil shipments to EU delayed 3 weeks.' },
  { id: 'semiconductors_tw', label: 'Semiconductors (TSMC)', category: 'raw_material', region: 'Taiwan', baseRisk: 0.65, description: 'Chip shipments to EU auto plants delayed — already in shortage crisis.' },
  { id: 'cotton_india', label: 'Cotton (India)', category: 'raw_material', region: 'India', baseRisk: 0.55, description: 'Indian cotton exports to Turkey/EU via Suez disrupted.' },
  { id: 'coffee_beans', label: 'Coffee Beans (Vietnam)', category: 'raw_material', region: 'Vietnam', baseRisk: 0.50, description: 'Robusta coffee shipments delayed, futures rose 3%.' },

  // Manufacturers
  { id: 'vw_wolfsburg', label: 'Volkswagen (Wolfsburg)', category: 'manufacturer', region: 'Germany', baseRisk: 0.58, description: 'Auto production slowed due to delayed Asian components.' },
  { id: 'toyota_europe', label: 'Toyota (EU Plants)', category: 'manufacturer', region: 'Europe', baseRisk: 0.55, description: 'Just-in-time delivery disrupted — temporary production halt.' },
  { id: 'basf_ludwigshafen', label: 'BASF Chemicals', category: 'manufacturer', region: 'Germany', baseRisk: 0.52, description: 'Chemical feedstock from Middle East delayed.' },
  { id: 'samsung_vietnam', label: 'Samsung (Vietnam)', category: 'manufacturer', region: 'Vietnam', baseRisk: 0.48, description: 'Electronics manufacturing — component shipping routes disrupted.' },
  { id: 'airbus_toulouse', label: 'Airbus (Toulouse)', category: 'manufacturer', region: 'France', baseRisk: 0.50, description: 'Aerospace components from Asia via Suez delayed.' },

  // Assembly & Distribution
  { id: 'amazon_eu', label: 'Amazon EU Fulfillment', category: 'assembly', region: 'Europe', baseRisk: 0.45, description: 'E-commerce inventory replenishment delayed by 2+ weeks.' },
  { id: 'ikea_supply', label: 'IKEA Supply Chain', category: 'assembly', region: 'Europe', baseRisk: 0.42, description: 'Furniture components from Asia delayed — stock shortages.' },
  { id: 'zara_inditex', label: 'Zara / Inditex', category: 'assembly', region: 'Spain', baseRisk: 0.40, description: 'Fast fashion supply chain — Asian fabric shipments delayed.' },

  // End-market logistics
  { id: 'tesco_uk', label: 'Tesco Distribution', category: 'logistics', region: 'UK', baseRisk: 0.38, description: 'Supermarket supply chain — imported goods delayed.' },
  { id: 'eu_energy_grid', label: 'EU Energy Grid', category: 'logistics', region: 'Europe', baseRisk: 0.70, description: 'Natural gas & oil supply disruption impacted energy prices.' },
  { id: 'retail_consumers', label: 'Global Retail (Consumers)', category: 'retail', region: 'Global', baseRisk: 0.30, description: 'End consumers — product shortages, price increases.' },
];

const SUEZ_LINKS: SuezLink[] = [
  // Canal → Ports
  { source: 'suez_canal', target: 'port_said', relation: 'blocks', weight: 1.0 },
  { source: 'suez_canal', target: 'port_suez', relation: 'blocks', weight: 1.0 },
  { source: 'suez_canal', target: 'jebel_ali', relation: 'congests', weight: 0.8 },
  { source: 'port_said', target: 'rotterdam', relation: 'delays_to', weight: 0.7 },
  { source: 'port_said', target: 'felixstowe', relation: 'delays_to', weight: 0.65 },
  { source: 'port_suez', target: 'singapore', relation: 'reroutes_via', weight: 0.6 },
  { source: 'port_suez', target: 'shanghai', relation: 'delays_from', weight: 0.55 },
  { source: 'jebel_ali', target: 'la_long_beach', relation: 'diverts_to', weight: 0.5 },

  // Shipping lines
  { source: 'suez_canal', target: 'evergreen', relation: 'blocked_by', weight: 1.0 },
  { source: 'suez_canal', target: 'maersk', relation: 'disrupts', weight: 0.8 },
  { source: 'suez_canal', target: 'cosco', relation: 'disrupts', weight: 0.75 },
  { source: 'maersk', target: 'rotterdam', relation: 'serves', weight: 0.6 },
  { source: 'cosco', target: 'shanghai', relation: 'serves', weight: 0.55 },
  { source: 'evergreen', target: 'singapore', relation: 'serves', weight: 0.5 },

  // Raw materials → Canal
  { source: 'crude_oil_gulf', target: 'suez_canal', relation: 'transits_via', weight: 0.85 },
  { source: 'lng_qatar', target: 'suez_canal', relation: 'transits_via', weight: 0.8 },
  { source: 'steel_asia', target: 'suez_canal', relation: 'transits_via', weight: 0.6 },
  { source: 'semiconductors_tw', target: 'suez_canal', relation: 'transits_via', weight: 0.65 },
  { source: 'cotton_india', target: 'suez_canal', relation: 'transits_via', weight: 0.55 },
  { source: 'coffee_beans', target: 'suez_canal', relation: 'transits_via', weight: 0.5 },

  // Ports → Manufacturers
  { source: 'rotterdam', target: 'vw_wolfsburg', relation: 'supplies', weight: 0.6 },
  { source: 'rotterdam', target: 'basf_ludwigshafen', relation: 'supplies', weight: 0.55 },
  { source: 'rotterdam', target: 'airbus_toulouse', relation: 'supplies', weight: 0.5 },
  { source: 'felixstowe', target: 'tesco_uk', relation: 'supplies', weight: 0.5 },
  { source: 'shanghai', target: 'samsung_vietnam', relation: 'feeds', weight: 0.45 },
  { source: 'rotterdam', target: 'toyota_europe', relation: 'supplies', weight: 0.55 },

  // Raw materials → Manufacturers
  { source: 'crude_oil_gulf', target: 'basf_ludwigshafen', relation: 'feedstock', weight: 0.7 },
  { source: 'crude_oil_gulf', target: 'eu_energy_grid', relation: 'fuels', weight: 0.75 },
  { source: 'lng_qatar', target: 'eu_energy_grid', relation: 'fuels', weight: 0.7 },
  { source: 'semiconductors_tw', target: 'vw_wolfsburg', relation: 'component', weight: 0.65 },
  { source: 'semiconductors_tw', target: 'samsung_vietnam', relation: 'component', weight: 0.6 },
  { source: 'steel_asia', target: 'vw_wolfsburg', relation: 'material', weight: 0.5 },
  { source: 'cotton_india', target: 'zara_inditex', relation: 'material', weight: 0.5 },

  // Manufacturers → Assembly/Retail
  { source: 'vw_wolfsburg', target: 'retail_consumers', relation: 'sells_to', weight: 0.4 },
  { source: 'toyota_europe', target: 'retail_consumers', relation: 'sells_to', weight: 0.4 },
  { source: 'samsung_vietnam', target: 'amazon_eu', relation: 'ships_to', weight: 0.5 },
  { source: 'basf_ludwigshafen', target: 'retail_consumers', relation: 'industrial', weight: 0.35 },
  { source: 'amazon_eu', target: 'retail_consumers', relation: 'delivers_to', weight: 0.45 },
  { source: 'ikea_supply', target: 'retail_consumers', relation: 'delivers_to', weight: 0.4 },
  { source: 'zara_inditex', target: 'retail_consumers', relation: 'delivers_to', weight: 0.4 },
  { source: 'tesco_uk', target: 'retail_consumers', relation: 'delivers_to', weight: 0.4 },
  { source: 'eu_energy_grid', target: 'retail_consumers', relation: 'powers', weight: 0.5 },
];

// Hyperedges: groups of nodes that share correlated disruption risk
const SUEZ_HYPEREDGES: SuezHyperedge[] = [
  { id: 'he_canal_zone', name: 'Suez Corridor Blockage Zone', nodes: ['suez_canal', 'port_said', 'port_suez', 'jebel_ali', 'evergreen'], type: 'geographic_corridor', color: '#EF4444' },
  { id: 'he_energy', name: 'Energy Supply Chain', nodes: ['crude_oil_gulf', 'lng_qatar', 'eu_energy_grid', 'basf_ludwigshafen'], type: 'commodity_chain', color: '#F59E0B' },
  { id: 'he_auto', name: 'Automotive Supply Chain', nodes: ['semiconductors_tw', 'steel_asia', 'vw_wolfsburg', 'toyota_europe', 'retail_consumers'], type: 'manufacturing_chain', color: '#3B82F6' },
  { id: 'he_eu_ports', name: 'EU Port Congestion Cluster', nodes: ['rotterdam', 'felixstowe', 'maersk', 'cosco'], type: 'logistics_cluster', color: '#8B5CF6' },
  { id: 'he_consumer', name: 'Consumer Goods Chain', nodes: ['cotton_india', 'coffee_beans', 'amazon_eu', 'ikea_supply', 'zara_inditex', 'tesco_uk', 'retail_consumers'], type: 'retail_chain', color: '#10B981' },
  { id: 'he_asia_route', name: 'Asia–Europe Shipping Lane', nodes: ['shanghai', 'singapore', 'suez_canal', 'rotterdam', 'felixstowe', 'maersk', 'cosco', 'evergreen'], type: 'shipping_route', color: '#06B6D4' },
];

// Real timeline data from the March 2021 Ever Given incident
const SUEZ_TIMELINE = [
  { day: 'Mar 23', year: '2021', event: 'Ever Given runs aground — Canal blocked', severity: 1.0, shipsBlocked: 0, tradeBlocked: 0,
    detail: 'The 400m container ship Ever Given loses steering in a sandstorm at 07:40 local time, wedging diagonally across the canal. All north- and south-bound traffic halts immediately.',
    impacts: ['Canal 100% blocked', 'Brent crude jumps 2%', '12% of global trade halted'],
    phase: 'crisis' as const },
  { day: 'Mar 24', year: '2021', event: 'Tugboats attempt to dislodge — backlog grows', severity: 0.95, shipsBlocked: 150, tradeBlocked: 9.6,
    detail: '11 tugboats work around the clock attempting to free the 220,000-ton vessel. The Suez Canal Authority (SCA) suspends all transit. Ships begin queuing at both canal entrances.',
    impacts: ['150 ships queued', '$9.6B/day trade blocked', 'Insurance claims surge'],
    phase: 'crisis' as const },
  { day: 'Mar 25', year: '2021', event: 'Dredging operations begin — oil prices spike 6%', severity: 0.92, shipsBlocked: 250, tradeBlocked: 19.2,
    detail: 'Boskalis subsidiary SMIT Salvage joins the operation. Giant dredger Mashoor begins removing 30,000 cubic meters of sand. Brent crude hits $64/barrel (+6%). LNG spot prices in Asia spike 10%.',
    impacts: ['Oil +6%, LNG +10%', '250 ships waiting', 'Just-in-time supply chains disrupted'],
    phase: 'crisis' as const },
  { day: 'Mar 26', year: '2021', event: 'Cape of Good Hope rerouting begins — +12 days per voyage', severity: 0.90, shipsBlocked: 320, tradeBlocked: 28.8,
    detail: 'Maersk and Hapag-Lloyd announce rerouting via the Cape of Good Hope, adding 6,000 nautical miles and $300K+ fuel cost per ship. Transit time increases from 12 to 24 days.',
    impacts: ['+12 days per voyage', '+$300K fuel/ship', '320 ships in queue'],
    phase: 'crisis' as const },
  { day: 'Mar 27', year: '2021', event: 'Container rates surge 47% — global supply alarm', severity: 0.88, shipsBlocked: 369, tradeBlocked: 38.4,
    detail: 'The Drewry World Container Index jumps 47%. Auto manufacturers in Europe warn of parts shortages within days. IKEA reports significant delays. US retailers brace for spring inventory shortfalls.',
    impacts: ['Container rates +47%', 'Auto parts shortages', 'Retail delays worldwide'],
    phase: 'crisis' as const },
  { day: 'Mar 28', year: '2021', event: 'Dredging progress — partial refloating attempted', severity: 0.82, shipsBlocked: 422, tradeBlocked: 48.0,
    detail: 'High spring tide provides opportunity. The stern is partially refloated for the first time. 27,000 cubic meters of sand removed. Japanese owner Shoei Kisen apologizes publicly. 422 ships now waiting.',
    impacts: ['Stern partially freed', '422 ships blocked', '$48B cumulative trade impact'],
    phase: 'crisis' as const },
  { day: 'Mar 29', year: '2021', event: 'Ever Given freed — Canal reopens', severity: 0.40, shipsBlocked: 450, tradeBlocked: 54.0,
    detail: 'At 15:05 local time, Ever Given is fully refloated on a spring high tide after 6 days. The ship is moved to Great Bitter Lake for inspection. SCA begins processing the backlog of 450 vessels.',
    impacts: ['Canal reopened after 6 days', '450 ships to clear', 'Global markets rally'],
    phase: 'recovery' as const },
  { day: 'Apr 3', year: '2021', event: 'Backlog clearing — congestion persists at ports', severity: 0.55, shipsBlocked: 200, tradeBlocked: 54.0,
    detail: 'SCA works 24/7, clearing ~80 ships/day. However, the simultaneous arrival of hundreds of vessels at destination ports (Rotterdam, Singapore, Felixstowe) creates secondary congestion waves.',
    impacts: ['Port congestion cascades', 'Container dwell time +40%', 'Warehouse overflow'],
    phase: 'recovery' as const },
  { day: 'Apr 10', year: '2021', event: 'Rotterdam congestion peaks — 2 week delays', severity: 0.50, shipsBlocked: 50, tradeBlocked: 54.0,
    detail: 'Port of Rotterdam reports record congestion with berthing delays of 7-14 days. Container terminals at capacity. Trucking companies face 3-day wait times. Empty container repositioning breaks down globally.',
    impacts: ['Rotterdam: 14-day delays', 'Container repositioning crisis', 'Trucking backlogs'],
    phase: 'ripple' as const },
  { day: 'Apr 20', year: '2021', event: 'Ripple effects continue — container shortage globally', severity: 0.35, shipsBlocked: 0, tradeBlocked: 54.0,
    detail: 'Although the canal is clear, the disruption amplified the existing global container shortage. Containers are in the wrong locations. Factory output in China exceeds available shipping capacity. Semiconductor deliveries delayed further.',
    impacts: ['Global container shortage', 'Factory output backlogged', 'Chip shortage worsened'],
    phase: 'ripple' as const },
  { day: 'May 1', year: '2021', event: 'Lasting impact — freight rates remain 30% elevated', severity: 0.25, shipsBlocked: 0, tradeBlocked: 54.0,
    detail: 'Six weeks later, freight rates remain 30% above pre-blockage levels. The incident exposed critical single-point-of-failure vulnerabilities in global supply chains, accelerating supply chain diversification strategies and digital twin adoption.',
    impacts: ['Freight rates +30% sustained', 'Supply chain redesign accelerated', 'Total cost: ~$54B+ trade disrupted'],
    phase: 'aftermath' as const },
];

const CATEGORY_COLORS: Record<SuezNodeCategory, string> = {
  canal: '#EF4444',
  port: '#F59E0B',
  shipping: '#06B6D4',
  raw_material: '#8B5CF6',
  manufacturer: '#3B82F6',
  assembly: '#10B981',
  logistics: '#F97316',
  retail: '#EC4899',
};

const CATEGORY_LABELS: Record<SuezNodeCategory, string> = {
  canal: 'Canal Chokepoint',
  port: 'Major Port',
  shipping: 'Shipping Line',
  raw_material: 'Raw Material',
  manufacturer: 'Manufacturer',
  assembly: 'Assembly / Retail',
  logistics: 'Logistics Hub',
  retail: 'End Market',
};

const PREDICTION_STEPS = [
  {
    icon: FileSpreadsheet,
    title: '1. Data Ingestion',
    desc: 'Raw supply chain data (orders, shipments, components) is loaded and normalized through the DataAdapter. Numeric features are scaled to [0,1], categorical features are encoded, and timestamps are aligned to a unified temporal grid.',
    color: 'from-blue-500 to-blue-600',
    colorHex: '#3B82F6',
    details: [
      'Supports 5 datasets: DataCo (180K), BOM (12K), Ports (847), Maintenance (10K), Retail (30K)',
      'Missing values imputed with column medians; outliers clipped at 3σ',
      'Timestamps unified to daily granularity with forward-fill',
    ],
    visual: 'ingest' as const,
  },
  {
    icon: Network,
    title: '2. Hypergraph Construction',
    desc: 'Unlike standard graphs with pairwise edges, we build hyperedges that connect multiple related nodes simultaneously. A single hyperedge might connect a supplier + 3 manufacturers + 2 shipping routes that share risk.',
    color: 'from-cyan-500 to-cyan-600',
    colorHex: '#06B6D4',
    details: [
      'DynamicHyperedgeConstructor mines temporal co-occurrence patterns',
      'Incidence matrix H ∈ {0,1}^(N×E) maps nodes to hyperedges',
      'Hyperedge weights learned via DynamicHyperedgeWeightLearner',
    ],
    visual: 'hypergraph' as const,
  },
  {
    icon: Layers,
    title: '3. Spectral Convolution',
    desc: 'Node features propagate through the hypergraph using spectral convolution (Zhou et al. formulation). Each node aggregates information from ALL members of its hyperedges — not just direct neighbors.',
    color: 'from-violet-500 to-violet-600',
    colorHex: '#8B5CF6',
    details: [
      'Formula: X\' = D_v^(-½) H W D_e^(-1) H^T D_v^(-½) X Θ',
      'Residual connections + LayerNorm stabilize deep propagation',
      '2-layer spectral conv with 128-dim hidden features',
    ],
    visual: 'spectral' as const,
  },
  {
    icon: Activity,
    title: '4. Temporal Fusion',
    desc: 'A Bi-LSTM captures sequential patterns (seasonal trends, delivery cycles), while a Transformer encoder captures long-range dependencies. A learned gating mechanism fuses both streams.',
    color: 'from-amber-500 to-amber-600',
    colorHex: '#F59E0B',
    details: [
      'Bi-LSTM: 2 layers, 128 hidden, captures local patterns',
      'Transformer: 4 heads, 2 layers, captures long-range deps',
      'Gating: g = σ(W₁·LSTM + W₂·Transformer + b), output = g⊙LSTM + (1-g)⊙Transformer',
    ],
    visual: 'temporal' as const,
  },
  {
    icon: GitBranch,
    title: '5. Relation Attention',
    desc: 'Five heterogeneous relation types are processed with type-specific attention: supplier_of, manufactured_by, transported_by, quality_controlled_by, and co_disrupted_with.',
    color: 'from-emerald-500 to-emerald-600',
    colorHex: '#10B981',
    details: [
      'Softmax attention learns per-relation importance weights',
      'co_disrupted_with is a v2.0 addition for correlated failures',
      'Attention scores provide interpretable relation importance',
    ],
    visual: 'relation' as const,
  },
  {
    icon: Target,
    title: '6. Multi-Task Risk Output',
    desc: 'Four prediction heads output simultaneously: price change, disruption detection, criticality classification, and cascade risk scoring. Multi-task learning ensures a rich shared representation.',
    color: 'from-red-500 to-red-600',
    colorHex: '#EF4444',
    details: [
      'Price: MSE loss · Change: BCE loss · Criticality: 4-class CE',
      'Cascade risk: KL divergence loss on spread distribution',
      'Loss weighting: 1.0·price + 0.8·change + 1.2·criticality + 0.6·cascade',
    ],
    visual: 'output' as const,
  },
];

/* -----------------------------------------------------------------------
   1. UTILITY COMPONENTS
   ----------------------------------------------------------------------- */

const FadeIn: React.FC<{
  children: React.ReactNode;
  delay?: number;
  className?: string;
  direction?: 'up' | 'left' | 'right';
}> = ({ children, delay = 0, className = '', direction = 'up' }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: '-80px' });

  const initial = {
    opacity: 0,
    y: direction === 'up' ? 40 : 0,
    x: direction === 'left' ? -40 : direction === 'right' ? 40 : 0,
  };

  return (
    <motion.div
      ref={ref}
      initial={initial}
      animate={isInView ? { opacity: 1, y: 0, x: 0 } : initial}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={className}
    >
      {children}
    </motion.div>
  );
};

const Section: React.FC<{
  id: string;
  children: React.ReactNode;
  className?: string;
}> = ({ id, children, className = '' }) => (
  <section id={id} className={`relative py-24 md:py-32 ${className}`}>
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">{children}</div>
  </section>
);

const SectionHeading: React.FC<{ children: React.ReactNode; sub?: string }> = ({
  children,
  sub,
}) => (
  <FadeIn className="text-center mb-16">
    <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4 tracking-tight">
      {children}
    </h2>
    {sub && <p className="text-gray-400 text-lg max-w-2xl mx-auto">{sub}</p>}
    <div className="mt-6 mx-auto w-24 h-1 rounded-full bg-gradient-to-r from-accent-blue to-accent-cyan" />
  </FadeIn>
);

const GlassCard: React.FC<{
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  style?: React.CSSProperties;
}> = ({ children, className = '', hover = true, style }) => (
  <div
    className={`glass rounded-2xl p-6 ${hover ? 'glass-hover' : ''} ${className}`}
    style={style}
  >
    {children}
  </div>
);

const AnimatedCounter: React.FC<{
  value: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
}> = ({ value, suffix = '', prefix = '', decimals = 0 }) => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!isInView) return;
    const end = value;
    const duration = 2000;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * end);
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [isInView, value]);

  return (
    <span ref={ref} className="tabular-nums">
      {prefix}
      {decimals > 0 ? display.toFixed(decimals) : Math.round(display)}
      {suffix}
    </span>
  );
};

/* -----------------------------------------------------------------------
   2. NETWORK BACKGROUND (Canvas)
   ----------------------------------------------------------------------- */

interface NetNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

const NetworkBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<NetNode[]>([]);
  const animRef = useRef<number>(0);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    canvas.width = w;
    canvas.height = h;

    const count = Math.floor((w * h) / 18000);
    nodesRef.current = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.5 + 0.5,
    }));
  }, []);

  useEffect(() => {
    init();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const nodes = nodesRef.current;
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      }
      const maxDist = 120;
      ctx.lineWidth = 0.5;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < maxDist) {
            const alpha = (1 - d / maxDist) * 0.15;
            ctx.strokeStyle = `rgba(59,130,246,${alpha})`;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(59,130,246,0.5)';
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(draw);
    };
    draw();

    const onResize = () => init();
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', onResize);
    };
  }, [init]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ opacity: 0.6 }}
    />
  );
};

/* -----------------------------------------------------------------------
   3. NAVBAR
   ----------------------------------------------------------------------- */

const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setMobileOpen(false);
  };

  return (
    <motion.nav
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'py-3 bg-navy-900/80 backdrop-blur-xl border-b border-white/5 shadow-lg shadow-black/20'
          : 'py-5 bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        <button onClick={() => scrollTo('hero')} className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent-blue to-accent-cyan flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <span className="text-white font-bold text-lg tracking-tight hidden sm:inline">
            HT-HGNN
          </span>
        </button>

        <div className="hidden lg:flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollTo(item.id)}
              className="px-3 py-2 text-sm text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-white/5"
            >
              {item.label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="lg:hidden p-2 text-gray-400 hover:text-white"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:hidden bg-navy-900/95 backdrop-blur-xl border-t border-white/5 mt-2"
        >
          <div className="px-4 py-4 space-y-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                onClick={() => scrollTo(item.id)}
                className="block w-full text-left px-4 py-2.5 text-sm text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
              >
                {item.label}
              </button>
            ))}
          </div>
        </motion.div>
      )}
    </motion.nav>
  );
};

/* -----------------------------------------------------------------------
   4. HERO SECTION
   ----------------------------------------------------------------------- */

const HeroSection: React.FC = () => {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.15], [1, 0]);
  const y = useTransform(scrollYProgress, [0, 0.15], [0, -60]);

  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <NetworkBackground />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_20%,#0A0F1E_70%)]" />

      <motion.div style={{ opacity, y }} className="relative z-10 text-center px-4 max-w-5xl mx-auto">
        <FadeIn delay={0.2}>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass text-xs font-medium text-accent-cyan mb-8 border border-accent-cyan/20">
            <Zap className="w-3.5 h-3.5" />
            Final Year Engineering Project — 2026
          </div>
        </FadeIn>

        <FadeIn delay={0.4}>
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-white leading-[1.1] tracking-tight mb-6">
            Supply Chain Risk Analysis
            <br />
            <span className="gradient-text">using HT-HGNN</span>
          </h1>
        </FadeIn>

        <FadeIn delay={0.6}>
          <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Predicting disruptions before they happen using Heterogeneous Temporal
            Hypergraph Neural Networks with cascade simulation &amp; explainability.
          </p>
        </FadeIn>

        <FadeIn delay={0.8}>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="#try-it"
              className="inline-flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl bg-gradient-to-r from-accent-blue to-blue-600 text-white font-semibold text-sm hover:shadow-lg hover:shadow-accent-blue/25 transition-all duration-300 hover:-translate-y-0.5"
            >
              Try It Live
              <Upload className="w-4 h-4" />
            </a>
            <a
              href="#how-it-works"
              className="inline-flex items-center justify-center gap-2 px-8 py-3.5 rounded-xl border border-white/10 text-white font-semibold text-sm hover:bg-white/5 transition-all duration-300"
            >
              How It Predicts
              <Eye className="w-4 h-4" />
            </a>
          </div>
        </FadeIn>

        <FadeIn delay={1.0}>
          <p className="mt-16 text-sm text-gray-500">
            Department of Mechanical Engineering
          </p>
        </FadeIn>
      </motion.div>

      <motion.div
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 text-gray-500"
      >
        <ChevronDown className="w-6 h-6" />
      </motion.div>
    </section>
  );
};

/* -----------------------------------------------------------------------
   5. PROBLEM SECTION
   ----------------------------------------------------------------------- */

const ProblemSection: React.FC = () => (
  <Section id="problem" className="bg-grid">
    <SectionHeading sub="Why existing tools fall short for modern supply chains">
      The Problem
    </SectionHeading>

    <div className="grid md:grid-cols-3 gap-6">
      {PROBLEM_CARDS.map((card, i) => (
        <FadeIn key={i} delay={i * 0.15}>
          <GlassCard className={`h-full ${card.border} border`}>
            <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${card.color} flex items-center justify-center mb-5`}>
              <card.icon className="w-6 h-6 text-white" />
            </div>
            <div className="text-3xl font-bold text-white mb-2">{card.stat}</div>
            <p className="text-gray-400 leading-relaxed">{card.text}</p>
          </GlassCard>
        </FadeIn>
      ))}
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   6. SOLUTION SECTION
   ----------------------------------------------------------------------- */

const SolutionSection: React.FC = () => (
  <Section id="solution">
    <SectionHeading sub="A unified neural architecture for multi-way temporal supply chain modeling">
      Our Solution — HT-HGNN
    </SectionHeading>

    <div className="grid lg:grid-cols-2 gap-12 items-center">
      <FadeIn direction="left" className="relative">
        <div className="grid grid-cols-2 gap-6">
          <GlassCard hover={false} className="text-center">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
              Standard Graph
            </p>
            <svg viewBox="0 0 160 160" className="w-full max-w-[160px] mx-auto">
              <line x1="40" y1="40" x2="120" y2="50" stroke="#334155" strokeWidth="1.5" />
              <line x1="40" y1="40" x2="80" y2="130" stroke="#334155" strokeWidth="1.5" />
              <line x1="120" y1="50" x2="80" y2="130" stroke="#334155" strokeWidth="1.5" />
              <circle cx="40" cy="40" r="8" fill="#475569" />
              <circle cx="120" cy="50" r="8" fill="#475569" />
              <circle cx="80" cy="130" r="8" fill="#475569" />
            </svg>
            <p className="text-xs text-gray-500 mt-3">Pairwise edges only</p>
          </GlassCard>

          <GlassCard hover={false} className="text-center border-accent-blue/20">
            <p className="text-xs font-semibold text-accent-cyan uppercase tracking-wider mb-4">
              Hypergraph
            </p>
            <svg viewBox="0 0 160 160" className="w-full max-w-[160px] mx-auto">
              <path
                d="M30,50 Q80,10 130,50 Q140,100 80,140 Q20,110 30,50Z"
                fill="rgba(59,130,246,0.12)"
                stroke="#3B82F6"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />
              <circle cx="50" cy="55" r="8" fill="#3B82F6" />
              <circle cx="110" cy="55" r="8" fill="#3B82F6" />
              <circle cx="80" cy="120" r="8" fill="#06B6D4" />
              <circle cx="75" cy="75" r="5" fill="#06B6D4" opacity="0.7" />
              <circle cx="80" cy="80" r="50" fill="none" stroke="#3B82F6" strokeWidth="0.5" opacity="0.2" />
            </svg>
            <p className="text-xs text-accent-cyan mt-3">Multi-node hyperedge</p>
          </GlassCard>
        </div>
      </FadeIn>

      <FadeIn direction="right">
        <div className="space-y-6">
          {[
            {
              icon: Layers,
              title: 'Spectral Hypergraph Convolution',
              desc: 'Captures multi-way group interactions — a single hyperedge connects entire supplier clusters, production lines, or shipping corridors simultaneously.',
            },
            {
              icon: Activity,
              title: 'Temporal Fusion Encoder',
              desc: 'Bi-LSTM + Transformer architecture learns both short-term patterns and long-range temporal dependencies across supply chain events.',
            },
            {
              icon: GitBranch,
              title: 'Heterogeneous Relation Attention',
              desc: '5 distinct relation types (supplier, manufacturer, transporter, QC, co-disrupted) processed with type-specific attention weights.',
            },
          ].map((item, i) => (
            <div key={i} className="flex gap-4 items-start">
              <div className="w-10 h-10 rounded-lg bg-accent-blue/10 border border-accent-blue/20 flex items-center justify-center shrink-0 mt-0.5">
                <item.icon className="w-5 h-5 text-accent-blue" />
              </div>
              <div>
                <h3 className="text-white font-semibold mb-1">{item.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </FadeIn>
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   7. ARCHITECTURE SECTION
   ----------------------------------------------------------------------- */

const ArchitectureSection: React.FC = () => (
  <Section id="architecture" className="bg-grid">
    <SectionHeading sub="End-to-end pipeline from raw data to multi-task risk prediction">
      Model Architecture
    </SectionHeading>

    <div className="hidden lg:flex items-start justify-between gap-2">
      {ARCH_STEPS.map((step, i) => (
        <FadeIn key={i} delay={i * 0.12} className="flex items-start flex-1">
          <GlassCard className="w-full text-center border border-accent-blue/10">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-blue/20 to-accent-cyan/20 flex items-center justify-center mx-auto mb-4">
              <step.icon className="w-6 h-6 text-accent-blue" />
            </div>
            <h3 className="text-white font-semibold text-sm mb-2">{step.title}</h3>
            <p className="text-gray-500 text-xs leading-relaxed">{step.desc}</p>
          </GlassCard>

          {i < ARCH_STEPS.length - 1 && (
            <div className="flex items-center px-1 pt-12 shrink-0">
              <div className="w-8 h-[2px] bg-gradient-to-r from-accent-blue to-accent-cyan rounded-full" />
              <ArrowRight className="w-4 h-4 text-accent-blue -ml-1" />
            </div>
          )}
        </FadeIn>
      ))}
    </div>

    <div className="lg:hidden space-y-4">
      {ARCH_STEPS.map((step, i) => (
        <FadeIn key={i} delay={i * 0.1}>
          <GlassCard className="flex items-center gap-4 border border-accent-blue/10">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-blue/20 to-accent-cyan/20 flex items-center justify-center shrink-0">
              <step.icon className="w-6 h-6 text-accent-blue" />
            </div>
            <div>
              <h3 className="text-white font-semibold text-sm">{step.title}</h3>
              <p className="text-gray-500 text-xs">{step.desc}</p>
            </div>
          </GlassCard>
        </FadeIn>
      ))}
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   8. HOW IT PREDICTS — Step-by-step prediction flow
   ----------------------------------------------------------------------- */

const HowItWorksSection: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  // Mini visual components for each step
  const StepVisual: React.FC<{ type: string; active: boolean; color: string }> = ({ type, active, color }) => {
    if (type === 'ingest') {
      return (
        <div className="relative h-full flex flex-col items-center justify-center gap-4 py-4">
          <div className="flex items-end gap-2.5">
            {['CSV', 'JSON', 'SQL', 'API', 'IoT'].map((src, i) => (
              <motion.div
                key={src}
                className="flex flex-col items-center"
                initial={{ opacity: 0, y: -15 }}
                animate={active ? { opacity: 1, y: 0 } : { opacity: 0.3, y: -8 }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
              >
                <div className="text-xs text-gray-500 mb-1 font-medium">{src}</div>
                <div className="w-10 rounded-t" style={{ height: [56, 40, 48, 32, 24][i], backgroundColor: `${color}${active ? '40' : '15'}` }} />
              </motion.div>
            ))}
          </div>
          <motion.div
            className="flex items-center gap-1"
            animate={active ? { opacity: 1 } : { opacity: 0.3 }}
          >
            <div className="text-base text-gray-500">▼ ▼ ▼</div>
          </motion.div>
          <motion.div
            className="px-6 py-2.5 rounded-lg text-sm font-bold border"
            style={{ borderColor: `${color}30`, color, backgroundColor: `${color}10` }}
            animate={active ? { scale: [1, 1.05, 1] } : { scale: 1 }}
            transition={{ repeat: active ? Infinity : 0, duration: 2 }}
          >
            DataAdapter
          </motion.div>
          <div className="flex items-center gap-3 mt-1">
            {['Normalize [0,1]', 'Encode', 'Align ⏱'].map((t, i) => (
              <motion.span
                key={t}
                className="text-xs px-3 py-1 rounded-md border font-medium"
                style={{ borderColor: `${color}20`, color: `${color}CC` }}
                initial={{ opacity: 0 }}
                animate={active ? { opacity: 1 } : { opacity: 0.2 }}
                transition={{ delay: 0.5 + i * 0.15 }}
              >
                {t}
              </motion.span>
            ))}
          </div>
        </div>
      );
    }

    if (type === 'hypergraph') {
      const nodes = [
        { x: 25, y: 25 }, { x: 75, y: 18 }, { x: 55, y: 60 }, { x: 95, y: 55 }, { x: 35, y: 90 },
      ];
      return (
        <div className="h-full flex items-center justify-center gap-6 py-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-2 font-medium">Standard Graph</div>
            <svg width="160" height="130" viewBox="0 0 120 110" className="opacity-60">
              {[[0,1],[1,2],[2,3],[0,2],[3,4],[2,4]].map(([a,b], i) => (
                <line key={i} x1={nodes[a].x} y1={nodes[a].y} x2={nodes[b].x} y2={nodes[b].y} stroke="#64748B" strokeWidth="1.5" />
              ))}
              {nodes.map((n, i) => (
                <circle key={i} cx={n.x} cy={n.y} r="7" fill="#64748B" />
              ))}
              <text x="60" y="108" textAnchor="middle" fill="#64748B" fontSize="10">Pairwise only</text>
            </svg>
          </div>

          <motion.div
            className="text-gray-500 text-2xl font-bold"
            animate={active ? { x: [0, 5, 0], opacity: [0.4, 1, 0.4] } : { opacity: 0.3 }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >→</motion.div>

          <div className="text-center">
            <div className="text-xs mb-2 font-medium" style={{ color }}>Hypergraph</div>
            <svg width="160" height="130" viewBox="0 0 120 110">
              <motion.ellipse cx="50" cy="35" rx="45" ry="28" fill={`${color}15`} stroke={color} strokeWidth="1.2" strokeDasharray="4 3"
                animate={active ? { opacity: [0.4, 0.8, 0.4] } : { opacity: 0.2 }}
                transition={{ repeat: Infinity, duration: 2 }}
              />
              <motion.ellipse cx="65" cy="65" rx="40" ry="30" fill="#F59E0B15" stroke="#F59E0B" strokeWidth="1.2" strokeDasharray="4 3"
                animate={active ? { opacity: [0.3, 0.7, 0.3] } : { opacity: 0.2 }}
                transition={{ repeat: Infinity, duration: 2.5, delay: 0.3 }}
              />
              {nodes.map((n, i) => (
                <motion.circle key={i} cx={n.x} cy={n.y} r="8" fill={color}
                  animate={active ? { r: [7, 9, 7] } : { r: 6 }}
                  transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.2 }}
                />
              ))}
              <text x="60" y="108" textAnchor="middle" fill={color} fontSize="10">Multi-way edges</text>
            </svg>
          </div>
        </div>
      );
    }

    if (type === 'spectral') {
      return (
        <div className="h-full flex items-center justify-center py-4">
          <div className="flex items-center gap-3">
            <div className="text-center">
              <div className="text-xs text-gray-500 mb-2 font-medium">H (incidence)</div>
              <div className="grid grid-cols-3 gap-0.5">
                {[1,0,1, 1,1,0, 0,1,1, 1,0,0, 0,1,1, 1,1,0].map((v, i) => (
                  <motion.div
                    key={i}
                    className="w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-medium"
                    style={{ backgroundColor: v ? `${color}30` : 'rgba(255,255,255,0.03)', color: v ? color : '#374151' }}
                    animate={active ? { opacity: [0.5, 1, 0.5] } : { opacity: 0.3 }}
                    transition={{ delay: i * 0.03, repeat: active ? Infinity : 0, duration: 2 }}
                  >
                    {v}
                  </motion.div>
                ))}
              </div>
            </div>

            <motion.span className="text-gray-400 text-xl font-bold" animate={active ? { opacity: [0.3, 1, 0.3] } : { opacity: 0.3 }} transition={{ repeat: Infinity, duration: 1.5 }}>×</motion.span>

            <div className="text-center">
              <div className="text-xs text-gray-500 mb-2 font-medium">X (features)</div>
              <div className="grid grid-cols-2 gap-0.5">
                {[0.8, 0.3, 0.5, 0.9, 0.2, 0.7, 0.6, 0.4, 0.1, 0.8, 0.9, 0.2].map((v, i) => (
                  <motion.div
                    key={i}
                    className="w-9 h-7 rounded flex items-center justify-center text-xs font-mono"
                    style={{ backgroundColor: `rgba(139,92,246,${v * 0.4})`, color: '#C4B5FD' }}
                    animate={active ? { opacity: [0.5, 1, 0.5] } : { opacity: 0.3 }}
                    transition={{ delay: 0.3 + i * 0.03, repeat: active ? Infinity : 0, duration: 2 }}
                  >
                    {v.toFixed(1)}
                  </motion.div>
                ))}
              </div>
            </div>

            <motion.span className="text-gray-400 text-xl font-bold" animate={active ? { opacity: [0.3, 1, 0.3], x: [0, 3, 0] } : { opacity: 0.3 }} transition={{ repeat: Infinity, duration: 1.5 }}>→</motion.span>

            <div className="text-center">
              <div className="text-xs mb-2 font-medium" style={{ color }}>X&apos; (enriched)</div>
              <div className="grid grid-cols-2 gap-0.5">
                {[0.9, 0.6, 0.7, 0.8, 0.5, 0.9, 0.8, 0.5, 0.4, 0.9, 0.8, 0.6].map((v, i) => (
                  <motion.div
                    key={i}
                    className="w-9 h-7 rounded flex items-center justify-center text-xs font-mono font-bold"
                    style={{ backgroundColor: `${color}${Math.round(v * 60).toString(16).padStart(2, '0')}`, color }}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={active ? { opacity: 1, scale: 1 } : { opacity: 0.3, scale: 0.9 }}
                    transition={{ delay: 0.6 + i * 0.05 }}
                  >
                    {v.toFixed(1)}
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      );
    }

    if (type === 'temporal') {
      return (
        <div className="h-full flex flex-col items-center justify-center gap-3 py-4">
          <div className="flex items-end gap-1">
            {[0.3, 0.5, 0.4, 0.8, 0.6, 0.9, 0.7, 0.85, 0.95, 0.6, 0.4, 0.3].map((v, i) => (
              <motion.div
                key={i}
                className="w-4 rounded-t"
                style={{ height: v * 50, backgroundColor: `${color}${active ? '60' : '20'}` }}
                animate={active ? { height: v * 50, opacity: [0.5, 1, 0.5] } : { height: v * 35 }}
                transition={{ delay: i * 0.05, repeat: active ? Infinity : 0, duration: 2.5 }}
              />
            ))}
          </div>
          <div className="text-xs text-gray-500 font-medium">temporal input (t₁ ... t₁₂)</div>

          <div className="flex items-center gap-1 text-gray-500">▼ ▼</div>

          <div className="flex items-center gap-5">
            <motion.div
              className="px-5 py-2 rounded-lg text-sm font-bold border"
              style={{ borderColor: `${color}30`, color, backgroundColor: `${color}10` }}
              animate={active ? { boxShadow: [`0 0 0px ${color}00`, `0 0 12px ${color}40`, `0 0 0px ${color}00`] } : {}}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              Bi-LSTM
            </motion.div>
            <div className="text-lg text-gray-500 font-bold">+</div>
            <motion.div
              className="px-5 py-2 rounded-lg text-sm font-bold border"
              style={{ borderColor: '#F59E0B30', color: '#F59E0B', backgroundColor: '#F59E0B10' }}
              animate={active ? { boxShadow: ['0 0 0px #F59E0B00', '0 0 12px #F59E0B40', '0 0 0px #F59E0B00'] } : {}}
              transition={{ repeat: Infinity, duration: 2, delay: 0.5 }}
            >
              Transformer
            </motion.div>
          </div>

          <motion.div
            className="flex items-center gap-1"
            animate={active ? { opacity: [0.4, 1, 0.4] } : { opacity: 0.3 }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <span className="text-xs text-gray-400 font-medium">▼ gate σ(·) ▼</span>
          </motion.div>

          <motion.div
            className="px-6 py-2 rounded-lg text-sm font-bold"
            style={{ backgroundColor: `${color}20`, color }}
            animate={active ? { scale: [1, 1.05, 1] } : { scale: 1 }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            Fused Temporal Embedding
          </motion.div>
        </div>
      );
    }

    if (type === 'relation') {
      const relations = ['supplier_of', 'manufact.', 'transport.', 'QC', 'co_disrupted'];
      const weights = [0.25, 0.30, 0.15, 0.10, 0.20];
      return (
        <div className="h-full flex flex-col items-center justify-center gap-3 py-4">
          <div className="text-xs text-gray-500 font-medium">Relation Attention Weights</div>
          <div className="flex items-end gap-2.5">
            {relations.map((r, i) => (
              <div key={r} className="flex flex-col items-center gap-1.5">
                <motion.div
                  className="w-12 rounded-t text-[10px] flex items-end justify-center pb-1 font-mono font-medium"
                  style={{ backgroundColor: `${color}${Math.round(weights[i] * 200 + 40).toString(16)}`, color: '#fff' }}
                  initial={{ height: 0 }}
                  animate={active ? { height: weights[i] * 120 + 20 } : { height: weights[i] * 70 + 10 }}
                  transition={{ delay: i * 0.1, duration: 0.6 }}
                >
                  {active ? `${(weights[i] * 100).toFixed(0)}%` : ''}
                </motion.div>
                <span className="text-[9px] text-gray-500 -rotate-45 origin-top-left whitespace-nowrap font-medium">{r}</span>
              </div>
            ))}
          </div>
          <motion.div
            className="text-xs px-4 py-1.5 rounded-lg border font-medium mt-2"
            style={{ borderColor: `${color}30`, color }}
            animate={active ? { opacity: [0.5, 1, 0.5] } : { opacity: 0.3 }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            softmax → weighted fusion
          </motion.div>
        </div>
      );
    }

    // output type
    const outputs = [
      { label: 'Price Change', value: '$2.4K', sub: 'MSE', barColor: '#3B82F6', barW: 0.7 },
      { label: 'Disruption', value: '87%', sub: 'BCE', barColor: '#F59E0B', barW: 0.87 },
      { label: 'Criticality', value: 'HIGH', sub: 'CE', barColor: '#EF4444', barW: 0.85 },
      { label: 'Cascade Risk', value: '0.72', sub: 'KL', barColor: '#8B5CF6', barW: 0.72 },
    ];
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 py-4">
        {outputs.map((o, i) => (
          <div key={o.label} className="flex items-center gap-3 w-full px-4">
            <span className="text-xs text-gray-400 w-24 text-right font-medium">{o.label}</span>
            <div className="flex-1 h-4 bg-white/[0.04] rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: o.barColor }}
                initial={{ width: 0 }}
                animate={active ? { width: `${o.barW * 100}%` } : { width: `${o.barW * 50}%` }}
                transition={{ delay: i * 0.15, duration: 0.8 }}
              />
            </div>
            <motion.span
              className="text-sm font-mono font-bold w-14 text-right"
              style={{ color: o.barColor }}
              animate={active ? { opacity: 1 } : { opacity: 0.3 }}
            >
              {o.value}
            </motion.span>
          </div>
        ))}
        <div className="text-xs text-gray-500 mt-1 font-medium">
          Loss = 1.0·Price + 0.8·Change + 1.2·Criticality + 0.6·Cascade
        </div>
      </div>
    );
  };

  return (
    <Section id="how-it-works">
      <SectionHeading sub="A detailed walkthrough of how HT-HGNN predicts supply chain disruptions">
        How Disruption Prediction Works
      </SectionHeading>

      {/* Pipeline flow indicator */}
      <FadeIn className="mb-12">
        <div className="flex items-center justify-center gap-1 overflow-x-auto pb-2">
          {PREDICTION_STEPS.map((step, i) => (
            <div key={i} className="flex items-center gap-1">
              <button
                onClick={() => setActiveStep(i)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-all text-xs font-medium whitespace-nowrap ${
                  activeStep === i
                    ? 'bg-white/[0.08] border border-white/10 text-white shadow-lg'
                    : 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]'
                }`}
              >
                <step.icon className="w-3.5 h-3.5" style={{ color: activeStep === i ? step.colorHex : undefined }} />
                <span className="hidden sm:inline">{step.title.split('. ')[1]}</span>
                <span className="sm:hidden">{i + 1}</span>
              </button>
              {i < PREDICTION_STEPS.length - 1 && (
                <motion.div
                  className="w-4 h-px"
                  style={{ backgroundColor: i < activeStep ? PREDICTION_STEPS[i].colorHex : '#1E293B' }}
                  animate={i < activeStep ? { opacity: [0.5, 1, 0.5] } : { opacity: 0.3 }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              )}
            </div>
          ))}
        </div>
      </FadeIn>

      {/* Main content: alternating layout with visual */}
      <div className="relative">
        {/* Vertical line connector */}
        <div className="absolute left-6 lg:left-1/2 lg:-translate-x-px top-0 bottom-0 w-0.5 bg-gradient-to-b from-accent-blue via-accent-cyan to-accent-blue/20 hidden md:block" />

        <div className="space-y-8 md:space-y-10">
          {PREDICTION_STEPS.map((step, i) => {
            const isActive = activeStep === i;
            return (
              <FadeIn key={i} delay={i * 0.08} direction={i % 2 === 0 ? 'left' : 'right'}>
                <div
                  className={`flex flex-col md:flex-row items-stretch gap-5 cursor-pointer ${
                    i % 2 === 1 ? 'md:flex-row-reverse' : ''
                  }`}
                  onClick={() => setActiveStep(i)}
                >
                  {/* Content card */}
                  <div className={`flex-1 ${i % 2 === 1 ? 'md:text-right' : ''}`}>
                    <GlassCard
                      hover={false}
                      className={`border relative overflow-hidden h-full transition-all duration-300 ${
                        isActive ? 'border-white/10 shadow-lg' : 'border-white/5'
                      }`}
                    >
                      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${step.color}`} />
                      <div className={`flex items-center gap-3 mb-3 ${i % 2 === 1 ? 'md:flex-row-reverse' : ''}`}>
                        <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${step.color} flex items-center justify-center shrink-0`}>
                          <step.icon className="w-5 h-5 text-white" />
                        </div>
                        <h3 className="text-white font-bold text-lg">{step.title}</h3>
                      </div>
                      <p className="text-gray-400 text-sm leading-relaxed mb-3">{step.desc}</p>

                      {/* Technical detail bullets */}
                      <AnimatePresence>
                        {isActive && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                          >
                            <div className="pt-3 mt-3 border-t border-white/5 space-y-2">
                              {step.details.map((d, j) => (
                                <motion.div
                                  key={j}
                                  className={`flex items-start gap-2 ${i % 2 === 1 ? 'md:flex-row-reverse md:text-right' : ''}`}
                                  initial={{ opacity: 0, x: i % 2 === 0 ? -10 : 10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: j * 0.1 }}
                                >
                                  <span className="text-[10px] mt-0.5 shrink-0" style={{ color: step.colorHex }}>●</span>
                                  <span className="text-gray-400 text-xs leading-relaxed font-mono">{d}</span>
                                </motion.div>
                              ))}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </GlassCard>
                  </div>

                  {/* Center dot */}
                  <div className="hidden md:flex items-center justify-center w-10 shrink-0">
                    <motion.div
                      className={`rounded-full bg-gradient-to-br ${step.color}`}
                      animate={isActive ? { width: 18, height: 18, boxShadow: `0 0 12px ${step.colorHex}60` } : { width: 14, height: 14, boxShadow: '0 0 0px transparent' }}
                      style={{ outline: '4px solid #0A0F1E' }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>

                  {/* Visual panel */}
                  <div className="flex-1">
                    <GlassCard
                      hover={false}
                      className={`border h-full min-h-[280px] transition-all duration-300 ${
                        isActive ? 'border-white/10' : 'border-white/5 opacity-50'
                      }`}
                    >
                      <StepVisual type={step.visual} active={isActive} color={step.colorHex} />
                    </GlassCard>
                  </div>
                </div>
              </FadeIn>
            );
          })}
        </div>

        {/* Final output visualization */}
        <FadeIn delay={0.3} className="mt-16">
          <GlassCard hover={false} className="border border-accent-blue/20">
            <div className="text-center mb-6">
              <h3 className="text-white font-bold text-xl mb-2">Final Output — 4 Simultaneous Predictions</h3>
              <p className="text-gray-400 text-sm">Every node in the supply chain receives all four risk assessments in a single forward pass</p>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: 'Price Change', icon: TrendingUp, color: '#3B82F6', desc: 'Predicted price movement', metric: '±$2.4K', loss: 'MSE', example: 'Steel coil price drops 12% in 3 days' },
                { label: 'Disruption?', icon: AlertTriangle, color: '#F59E0B', desc: 'Binary disruption signal', metric: '87.2%', loss: 'BCE', example: 'P(disruption) for Port of Rotterdam' },
                { label: 'Criticality', icon: Shield, color: '#EF4444', desc: 'Low / Med / High / Critical', metric: 'HIGH', loss: '4-class CE', example: 'Suez Canal: CRITICAL (0.97 conf)' },
                { label: 'Cascade Risk', icon: Workflow, color: '#8B5CF6', desc: 'Spread probability score', metric: '0.72', loss: 'KL Div', example: 'Risk spreads to 18 downstream nodes' },
              ].map((out, i) => (
                <motion.div
                  key={i}
                  className="glass rounded-xl p-4 border border-white/5 relative overflow-hidden group"
                  whileHover={{ scale: 1.02, borderColor: `${out.color}30` }}
                >
                  <div className="absolute top-0 left-0 right-0 h-0.5" style={{ backgroundColor: out.color }} />
                  <out.icon className="w-6 h-6 mx-auto mb-2" style={{ color: out.color }} />
                  <p className="text-white font-semibold text-sm text-center">{out.label}</p>
                  <p className="text-gray-500 text-[10px] text-center mt-0.5">{out.desc}</p>
                  <div className="mt-3 pt-2 border-t border-white/5">
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] text-gray-600">Sample</span>
                      <span className="text-xs font-mono font-bold" style={{ color: out.color }}>{out.metric}</span>
                    </div>
                    <p className="text-[9px] text-gray-500 mt-1 leading-relaxed">{out.example}</p>
                    <span className="text-[8px] text-gray-600 mt-1 inline-block px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/5">Loss: {out.loss}</span>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Data flow summary */}
            <div className="mt-6 pt-4 border-t border-white/5">
              <div className="flex items-center justify-center gap-2 flex-wrap">
                {PREDICTION_STEPS.map((s, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <div className={`w-6 h-6 rounded bg-gradient-to-br ${s.color} flex items-center justify-center`}>
                      <s.icon className="w-3 h-3 text-white" />
                    </div>
                    {i < PREDICTION_STEPS.length - 1 && (
                      <motion.div
                        className="text-gray-600 text-xs"
                        animate={{ opacity: [0.3, 0.8, 0.3], x: [0, 2, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5, delay: i * 0.2 }}
                      >→</motion.div>
                    )}
                  </div>
                ))}
                <motion.div
                  className="text-gray-600 text-xs ml-1"
                  animate={{ opacity: [0.3, 0.8, 0.3] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                >=</motion.div>
                <span className="text-[10px] font-bold text-white bg-gradient-to-r from-accent-blue to-accent-cyan bg-clip-text text-transparent">
                  4 Risk Scores per Node
                </span>
              </div>
            </div>
          </GlassCard>
        </FadeIn>
      </div>
    </Section>
  );
};

/* -----------------------------------------------------------------------
   9. DATASETS SECTION — Expandable detail cards with links
   ----------------------------------------------------------------------- */

const DatasetDetailCard: React.FC<{ ds: typeof DATASETS_FULL[0]; index: number }> = ({ ds, index }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <FadeIn delay={index * 0.1}>
      <GlassCard className="border overflow-hidden" style={{ borderColor: `${ds.color}20` }} hover={false}>
        {/* Header — always visible */}
        <div className="flex items-start gap-4">
          <div
            className="w-14 h-14 rounded-xl flex items-center justify-center shrink-0"
            style={{ backgroundColor: `${ds.color}15` }}
          >
            <ds.icon className="w-7 h-7" style={{ color: ds.color }} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 flex-wrap">
              <h3 className="text-white font-bold text-lg">{ds.name}</h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded-full" style={{ backgroundColor: `${ds.color}15`, color: ds.color }}>
                {ds.id}
              </span>
            </div>
            <div className="flex flex-wrap gap-4 mt-2">
              <span className="text-sm text-gray-400"><span className="text-white font-semibold">{ds.stat}</span> records</span>
              <span className="text-sm text-gray-400">{ds.domain}</span>
              <span className="text-sm text-gray-400">{ds.timeSpan}</span>
            </div>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="shrink-0 w-9 h-9 rounded-lg border border-white/10 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/5 transition-all"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>

        {/* Expanded details */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className="mt-6 pt-6 border-t border-white/5 space-y-5">
                <div>
                  <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-2">Description</h4>
                  <p className="text-gray-400 text-sm leading-relaxed">{ds.description}</p>
                </div>

                <div className="grid md:grid-cols-2 gap-5">
                  <div>
                    <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-2">Features per Node</h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{ds.features}</p>
                  </div>
                  <div>
                    <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-2">Hyperedge Construction</h4>
                    <p className="text-gray-400 text-sm leading-relaxed">{ds.hyperedges}</p>
                  </div>
                </div>

                <div>
                  <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-2">Key Risk Factors Modeled</h4>
                  <p className="text-gray-400 text-sm leading-relaxed">{ds.riskFactors}</p>
                </div>

                <div className="flex items-center gap-4 pt-2">
                  <a
                    href={ds.sourceUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border"
                    style={{ borderColor: `${ds.color}30`, color: ds.color }}
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                    View Source Dataset
                  </a>
                  <span className="text-gray-600 text-xs">{ds.records} total records</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </GlassCard>
    </FadeIn>
  );
};

const DatasetsSection: React.FC = () => (
  <Section id="datasets" className="bg-grid">
    <SectionHeading sub="Validated on 5 diverse real-world supply chain datasets — click to explore each">
      Datasets In Detail
    </SectionHeading>

    <div className="space-y-5">
      {DATASETS_FULL.map((ds, i) => (
        <DatasetDetailCard key={ds.id} ds={ds} index={i} />
      ))}
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   10. TRY IT LIVE — Upload CSV + see predictions in charts
   ----------------------------------------------------------------------- */

interface PredictionResult {
  pricePredictions: number[];
  changePredictions: number[];
  criticalityScores: number[];
  nodeIds: string[];
}

const TryItLiveSection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.name.endsWith('.csv')) {
      setFile(dropped);
      setError(null);
      setResult(null);
    } else {
      setError('Please upload a CSV file');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const resp = await axios.post<PredictionResult>(`${API_BASE}/upload/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 30000,
      });

      setResult(resp.data);
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Upload failed. Make sure the backend is running on port 8000.';
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data from results
  const criticalityDistribution = result ? (() => {
    const bins = [
      { name: 'Low', range: [0, 0.25], count: 0, color: '#10B981' },
      { name: 'Medium', range: [0.25, 0.5], count: 0, color: '#F59E0B' },
      { name: 'High', range: [0.5, 0.75], count: 0, color: '#F97316' },
      { name: 'Critical', range: [0.75, 1.01], count: 0, color: '#EF4444' },
    ];
    for (const score of result.criticalityScores) {
      for (const bin of bins) {
        if (score >= bin.range[0] && score < bin.range[1]) {
          bin.count++;
          break;
        }
      }
    }
    return bins;
  })() : [];

  const priceData = result ? result.nodeIds.map((id, i) => ({
    node: id,
    price: Math.round(result.pricePredictions[i] * 100) / 100,
    criticality: Math.round(result.criticalityScores[i] * 100) / 100,
  })).slice(0, 30) : [];

  const scatterData = result ? result.nodeIds.map((_, i) => ({
    price: Math.round(result.pricePredictions[i] * 100) / 100,
    criticality: Math.round(result.criticalityScores[i] * 100) / 100,
    change: Math.round(Math.abs(result.changePredictions[i]) * 100) / 100,
  })) : [];

  const topRiskNodes = result ? result.nodeIds.map((id, i) => ({
    id,
    criticality: result.criticalityScores[i],
    price: result.pricePredictions[i],
    change: result.changePredictions[i],
  })).sort((a, b) => b.criticality - a.criticality).slice(0, 10) : [];

  return (
    <Section id="try-it">
      <SectionHeading sub="Upload your supply chain CSV data and see HT-HGNN predictions in real time">
        Try It Live
      </SectionHeading>

      {/* Upload area */}
      <FadeIn>
        <GlassCard hover={false} className="border border-accent-blue/20 mb-8">
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            {/* Drop zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center hover:border-accent-blue/30 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="hidden"
              />
              <Upload className="w-10 h-10 text-gray-500 mx-auto mb-4" />
              <p className="text-white font-semibold mb-1">
                {file ? file.name : 'Drop CSV file here or click to browse'}
              </p>
              <p className="text-gray-500 text-xs">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Supply chain data with numeric features — any CSV works'}
              </p>
              {file && (
                <div className="mt-4 flex items-center justify-center gap-2 text-accent-cyan text-sm">
                  <CheckCircle2 className="w-4 h-4" />
                  File ready
                </div>
              )}
            </div>

            {/* Info + button */}
            <div>
              <h3 className="text-white font-bold text-lg mb-3 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-accent-blue" />
                How it works
              </h3>
              <ol className="space-y-3 text-gray-400 text-sm">
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-accent-blue/20 text-accent-blue text-xs font-bold flex items-center justify-center shrink-0">1</span>
                  Upload a CSV with numeric supply chain features (orders, shipments, components, etc.)
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-accent-blue/20 text-accent-blue text-xs font-bold flex items-center justify-center shrink-0">2</span>
                  The backend extracts numeric columns and runs HT-HGNN inference
                </li>
                <li className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-accent-blue/20 text-accent-blue text-xs font-bold flex items-center justify-center shrink-0">3</span>
                  See live charts: price predictions, criticality scores, disruption risk, and top at-risk nodes
                </li>
              </ol>

              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className="mt-6 w-full inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-accent-blue to-blue-600 text-white font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-accent-blue/25 transition-all duration-300"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Running Inference...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4" />
                    Run Prediction
                  </>
                )}
              </button>

              {error && (
                <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                  {error}
                </div>
              )}
            </div>
          </div>
        </GlassCard>
      </FadeIn>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Summary stats */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {[
                { label: 'Nodes Analyzed', value: result.nodeIds.length.toLocaleString(), color: 'text-accent-blue' },
                { label: 'Avg. Criticality', value: (result.criticalityScores.reduce((a, b) => a + b, 0) / result.criticalityScores.length).toFixed(3), color: 'text-amber-400' },
                { label: 'High Risk Nodes', value: result.criticalityScores.filter(s => s > 0.7).length.toString(), color: 'text-red-400' },
                { label: 'Avg. Price', value: `$${(result.pricePredictions.reduce((a, b) => a + b, 0) / result.pricePredictions.length).toFixed(0)}`, color: 'text-emerald-400' },
              ].map((stat, i) => (
                <GlassCard key={i} className="text-center border border-white/5">
                  <div className={`text-2xl font-bold ${stat.color} mb-1`}>{stat.value}</div>
                  <p className="text-gray-500 text-xs">{stat.label}</p>
                </GlassCard>
              ))}
            </div>

            {/* Charts grid */}
            <div className="grid lg:grid-cols-2 gap-6 mb-8">
              {/* Criticality Distribution Pie */}
              <GlassCard hover={false} className="border border-white/5">
                <h3 className="text-white font-semibold mb-1">Criticality Distribution</h3>
                <p className="text-gray-500 text-xs mb-4">How nodes are distributed across risk levels</p>
                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={criticalityDistribution}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        dataKey="count"
                        nameKey="name"
                        label={({ name, count }) => `${name}: ${count}`}
                        labelLine={{ stroke: '#475569' }}
                      >
                        {criticalityDistribution.map((entry, idx) => (
                          <Cell key={idx} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0F172A',
                          border: '1px solid rgba(59,130,246,0.2)',
                          borderRadius: 12,
                          color: '#fff',
                          fontSize: 12,
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>

              {/* Price Predictions Line Chart */}
              <GlassCard hover={false} className="border border-white/5">
                <h3 className="text-white font-semibold mb-1">Price Predictions</h3>
                <p className="text-gray-500 text-xs mb-4">Predicted prices for first 30 nodes</p>
                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={priceData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <defs>
                        <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="node" tick={{ fill: '#64748B', fontSize: 9 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                      <YAxis tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid rgba(59,130,246,0.2)', borderRadius: 12, color: '#fff', fontSize: 12 }} />
                      <Area type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={2} fill="url(#priceGrad)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>

              {/* Scatter: Price vs Criticality */}
              <GlassCard hover={false} className="border border-white/5">
                <h3 className="text-white font-semibold mb-1">Price vs Criticality Risk</h3>
                <p className="text-gray-500 text-xs mb-4">Each dot is a supply chain node — higher Y = higher disruption risk</p>
                <div className="h-[280px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis type="number" dataKey="price" name="Price" tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} label={{ value: 'Predicted Price', position: 'insideBottom', offset: -2, style: { fill: '#64748B', fontSize: 10 } }} />
                      <YAxis type="number" dataKey="criticality" name="Criticality" domain={[0, 1]} tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} label={{ value: 'Criticality', angle: -90, position: 'insideLeft', style: { fill: '#64748B', fontSize: 10 } }} />
                      <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid rgba(59,130,246,0.2)', borderRadius: 12, color: '#fff', fontSize: 12 }} cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter data={scatterData} fill="#3B82F6" fillOpacity={0.6} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>

              {/* Top risk nodes */}
              <GlassCard hover={false} className="border border-white/5">
                <h3 className="text-white font-semibold mb-1">Top 10 At-Risk Nodes</h3>
                <p className="text-gray-500 text-xs mb-4">Nodes with highest criticality scores — prioritize for intervention</p>
                <div className="space-y-2 overflow-y-auto max-h-[280px] pr-2">
                  {topRiskNodes.map((node, i) => (
                    <div key={i} className="flex items-center gap-3 p-2.5 rounded-lg bg-white/[0.02] border border-white/5">
                      <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                        i < 3 ? 'bg-red-500/20 text-red-400' : 'bg-white/5 text-gray-500'
                      }`}>
                        {i + 1}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-white text-sm font-mono truncate">{node.id}</p>
                      </div>
                      <div className="text-right shrink-0">
                        <p className={`text-sm font-bold ${
                          node.criticality > 0.75 ? 'text-red-400' :
                          node.criticality > 0.5 ? 'text-amber-400' : 'text-emerald-400'
                        }`}>
                          {(node.criticality * 100).toFixed(1)}%
                        </p>
                        <p className="text-gray-600 text-[10px]">criticality</p>
                      </div>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </div>

            {/* Interpretation box */}
            <GlassCard hover={false} className="border border-accent-cyan/20">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-accent-cyan mt-0.5 shrink-0" />
                <div>
                  <h3 className="text-white font-semibold mb-2">Interpreting the Results</h3>
                  <div className="text-gray-400 text-sm leading-relaxed space-y-2">
                    <p><strong className="text-white">Criticality Score (0–1):</strong> Represents the predicted risk level for each node. Scores above 0.75 indicate <span className="text-red-400 font-medium">Critical</span> nodes that require immediate attention. The model considers hypergraph neighborhood, temporal trends, and multi-relation context.</p>
                    <p><strong className="text-white">Price Predictions:</strong> Estimated unit prices based on supply chain conditions. Sudden spikes or drops signal potential disruption (supplier shortage → price increase, demand collapse → price drop).</p>
                    <p><strong className="text-white">Change Predictions:</strong> The model's confidence that a significant change event (disruption, demand shift, logistics failure) will occur for each node. Values near 0 suggest stability; values near ±1 suggest imminent change.</p>
                  </div>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </Section>
  );
};

/* -----------------------------------------------------------------------
   11. RESULTS SECTION — Expanded with detailed breakdown
   ----------------------------------------------------------------------- */

const ResultsSection: React.FC = () => {
  const [showDetail, setShowDetail] = useState<number | null>(null);

  return (
    <Section id="results" className="bg-grid">
      <SectionHeading sub="Benchmarked against 6 baselines across all 5 datasets with ablation studies">
        Performance Results
      </SectionHeading>

      {/* Stat counters with expandable details */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mb-10">
        {STATS.map((s, i) => (
          <FadeIn key={i} delay={i * 0.1}>
            <div onClick={() => setShowDetail(showDetail === i ? null : i)} className="cursor-pointer">
              <GlassCard className={`text-center border transition-colors ${showDetail === i ? 'border-accent-blue/40' : 'border-accent-blue/10'}`}>
                <div className="text-3xl md:text-4xl font-extrabold text-white mb-2">
                  <AnimatedCounter
                    value={s.value}
                    suffix={s.suffix}
                    prefix={s.prefix}
                    decimals={s.decimals}
                  />
                </div>
                <p className="text-gray-400 text-sm">{s.label}</p>
                <p className="text-gray-600 text-[10px] mt-2">Click for details</p>
              </GlassCard>
            </div>
          </FadeIn>
        ))}
      </div>

      {/* Expandable detail for selected stat */}
      <AnimatePresence>
        {showDetail !== null && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-10 overflow-hidden"
          >
            <GlassCard hover={false} className="border border-accent-blue/20">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-accent-blue shrink-0 mt-0.5" />
                <p className="text-gray-400 text-sm leading-relaxed">{STATS[showDetail].detail}</p>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Charts side by side */}
      <div className="grid lg:grid-cols-2 gap-6 mb-10">
        {/* Benchmark comparison */}
        <FadeIn>
          <GlassCard hover={false} className="border border-accent-blue/10">
            <h3 className="text-white font-semibold mb-2 text-lg">Criticality Accuracy Comparison</h3>
            <p className="text-gray-500 text-xs mb-6">Our model vs. 6 baselines — all trained on identical data splits</p>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={BENCHMARK_DATA} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="name" tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                  <YAxis domain={[50, 100]} tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid rgba(59,130,246,0.2)', borderRadius: 12, color: '#fff', fontSize: 12 }} cursor={{ fill: 'rgba(59,130,246,0.05)' }} />
                  <Bar dataKey="accuracy" radius={[6, 6, 0, 0]} maxBarSize={50}>
                    {BENCHMARK_DATA.map((_, idx) => (
                      <Cell key={idx} fill={idx === BENCHMARK_DATA.length - 1 ? '#3B82F6' : idx === BENCHMARK_DATA.length - 2 ? '#06B6D4' : '#1E293B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </FadeIn>

        {/* Ablation study */}
        <FadeIn delay={0.15}>
          <GlassCard hover={false} className="border border-accent-blue/10">
            <h3 className="text-white font-semibold mb-2 text-lg">Ablation Study</h3>
            <p className="text-gray-500 text-xs mb-6">Removing each component to measure its contribution</p>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ABLATION_DATA} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                  <XAxis type="number" domain={[75, 100]} tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} width={120} />
                  <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid rgba(59,130,246,0.2)', borderRadius: 12, color: '#fff', fontSize: 12 }} cursor={{ fill: 'rgba(59,130,246,0.05)' }} />
                  <Bar dataKey="accuracy" radius={[0, 6, 6, 0]} maxBarSize={30}>
                    {ABLATION_DATA.map((_, idx) => (
                      <Cell key={idx} fill={idx === 0 ? '#3B82F6' : idx === ABLATION_DATA.length - 1 ? '#475569' : '#06B6D4'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </FadeIn>
      </div>

      {/* Detailed analysis */}
      <FadeIn>
        <GlassCard hover={false} className="border border-accent-blue/10">
          <h3 className="text-white font-semibold text-lg mb-4">Detailed Analysis</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-3">Why HT-HGNN Outperforms</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Standard GNNs (GCN, GAT) can only model pairwise relationships, missing the multi-way dependencies inherent in supply chains. When a shipping corridor connects 5 suppliers to 3 manufacturers, a regular graph needs 15 edges — our hypergraph uses 1 hyperedge. This dramatically improves message passing efficiency and captures group risk that pairwise models miss entirely.
              </p>
            </div>
            <div>
              <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-3">Temporal Component Impact</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Removing the temporal fusion encoder drops accuracy by 6.4% (94.7% → 88.3%), the largest single-component impact. This confirms that temporal patterns — seasonal demand cycles, delivery time trends, disruption aftershocks — are critical signals that static models cannot capture. The Bi-LSTM handles short-term patterns while the Transformer captures long-range dependencies.
              </p>
            </div>
            <div>
              <h4 className="text-accent-cyan text-xs font-semibold uppercase tracking-wider mb-3">Multi-Task Learning Benefit</h4>
              <p className="text-gray-400 text-sm leading-relaxed">
                Training all 4 prediction heads jointly (price, disruption, criticality, cascade) improves each individual task vs. single-task training. The shared representation learns richer features — price patterns inform disruption detection, criticality helps cascade estimation. The cascade head alone adds 3.2% to criticality accuracy through implicit risk propagation learning.
              </p>
            </div>
          </div>
        </GlassCard>
      </FadeIn>
    </Section>
  );
};

/* -----------------------------------------------------------------------
   12. NOVELTY SECTION
   ----------------------------------------------------------------------- */

const NoveltySection: React.FC = () => (
  <Section id="novelty">
    <SectionHeading sub="Key contributions and relation to prior work">
      What Makes This Novel
    </SectionHeading>

    <div className="grid md:grid-cols-3 gap-6">
      {NOVELTY_ITEMS.map((item, i) => (
        <FadeIn key={i} delay={i * 0.15}>
          <GlassCard className="h-full border border-accent-blue/10 flex flex-col">
            <div className="w-10 h-10 rounded-lg bg-accent-blue/10 flex items-center justify-center mb-4">
              <span className="text-accent-blue font-bold text-lg">{i + 1}</span>
            </div>
            <h3 className="text-white font-semibold text-lg mb-3">{item.title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed flex-1 mb-4">{item.desc}</p>
            <p className="text-gray-600 text-xs italic border-t border-white/5 pt-3">
              {item.cite}
            </p>
          </GlassCard>
        </FadeIn>
      ))}
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   13. SUEZ CANAL DISRUPTION SIMULATION
   ----------------------------------------------------------------------- */

const SuezDisruptionSection: React.FC = () => {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedNode, setSelectedNode] = useState<SuezNode | null>(null);
  const [hoveredHyperedge, setHoveredHyperedge] = useState<string | null>(null);
  const graphRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [graphDims, setGraphDims] = useState({ width: 800, height: 480 });

  // Measure container so ForceGraph3D is centered
  useEffect(() => {
    const el = graphContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setGraphDims({ width: Math.round(width), height: Math.round(height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Current disruption severity based on timeline step
  const currentTimeline = SUEZ_TIMELINE[step];
  const severity = currentTimeline.severity;

  // Compute node risk for current step — cascading from canal outward
  const nodeRisks = useMemo(() => {
    const risks: Record<string, number> = {};
    for (const node of SUEZ_NODES) {
      // Risk = baseRisk * current severity * distance decay
      risks[node.id] = Math.min(1, node.baseRisk * severity * (0.6 + 0.4 * node.baseRisk));
    }
    return risks;
  }, [severity]);

  // Build 3D graph data
  const graphData = useMemo(() => {
    const nodes = SUEZ_NODES.map(n => {
      const risk = nodeRisks[n.id] || 0;
      const isInActiveHyperedge = hoveredHyperedge
        ? SUEZ_HYPEREDGES.find(h => h.id === hoveredHyperedge)?.nodes.includes(n.id)
        : false;
      return {
        id: n.id,
        name: n.label,
        category: n.category,
        region: n.region,
        risk,
        val: n.category === 'canal' ? 24 : n.category === 'port' ? 12 : n.category === 'shipping' ? 10 : 8,
        color: n.category === 'canal' ? '#FF2222'
          : isInActiveHyperedge
          ? SUEZ_HYPEREDGES.find(h => h.id === hoveredHyperedge)!.color
          : risk > 0.7 ? '#EF4444' : risk > 0.5 ? '#F59E0B' : risk > 0.3 ? '#3B82F6' : '#10B981',
        description: n.description,
        baseRisk: n.baseRisk,
      };
    });

    const links = SUEZ_LINKS.map(l => ({
      source: l.source,
      target: l.target,
      relation: l.relation,
      color: `rgba(255,255,255,${0.1 + l.weight * severity * 0.4})`,
      width: 0.5 + l.weight * severity * 3,
    }));

    return { nodes, links };
  }, [nodeRisks, severity, hoveredHyperedge]);

  // Playback controls
  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(prev => {
          if (prev >= SUEZ_TIMELINE.length - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 2000);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing]);

  const resetSimulation = () => {
    setPlaying(false);
    setStep(0);
    setSelectedNode(null);
  };

  // Count disrupted nodes
  const disruptedCount = SUEZ_NODES.filter(n => (nodeRisks[n.id] || 0) > 0.5).length;
  const criticalCount = SUEZ_NODES.filter(n => (nodeRisks[n.id] || 0) > 0.7).length;

  return (
    <Section id="suez-simulation">
      <SectionHeading sub="Real-world case: March 2021 Ever Given blockage — watch how a single chokepoint cascades across the global supply chain">
        🚢 Suez Canal Disruption Simulation
      </SectionHeading>

      {/* Key stats bar */}
      <FadeIn>
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-8">
          {[
            { label: 'Ships Blocked', value: currentTimeline.shipsBlocked.toLocaleString(), color: 'text-red-400', icon: Ship },
            { label: 'Trade Blocked ($B)', value: `$${currentTimeline.tradeBlocked.toFixed(1)}B`, color: 'text-amber-400', icon: Globe },
            { label: 'Disrupted Nodes', value: disruptedCount.toString(), color: 'text-orange-400', icon: AlertTriangle },
            { label: 'Critical Nodes', value: criticalCount.toString(), color: 'text-red-500', icon: Zap },
            { label: 'Severity', value: `${(severity * 100).toFixed(0)}%`, color: 'text-cyan-400', icon: Activity },
          ].map((s, i) => (
            <GlassCard key={i} hover={false} className="text-center border border-white/5 py-3 px-2">
              <s.icon className={`w-4 h-4 ${s.color} mx-auto mb-1`} />
              <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
              <p className="text-gray-500 text-[10px]">{s.label}</p>
            </GlassCard>
          ))}
        </div>
      </FadeIn>

      <div className="grid lg:grid-cols-4 gap-5 mb-8">
        {/* LEFT: 3D Hypergraph Visualization */}
        <FadeIn className="lg:col-span-2">
          <GlassCard hover={false} className="border border-white/5 p-0 overflow-hidden">
            <div className="flex items-center justify-between px-4 pt-4 pb-2">
              <div>
                <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                  <Network className="w-4 h-4 text-accent-blue" />
                  3D Hypergraph — Live Cascade Propagation
                </h3>
                <p className="text-gray-500 text-xs mt-0.5">Rotate: drag &middot; Zoom: scroll &middot; Click node for details</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPlaying(!playing)}
                  className="p-2 rounded-lg bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30 transition-colors"
                >
                  {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={resetSimulation}
                  className="p-2 rounded-lg bg-white/5 text-gray-400 hover:bg-white/10 transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div ref={graphContainerRef} className="h-[480px] w-full relative">
              <ForceGraph3D
                ref={graphRef}
                width={graphDims.width}
                height={graphDims.height}
                graphData={graphData}
                nodeLabel={(node: any) => `<div style="background:#0F172A;border:1px solid rgba(59,130,246,0.3);border-radius:8px;padding:8px 14px;color:white;font-size:12px;max-width:260px;line-height:1.5">
                  <strong style="font-size:13px">${node.name}</strong><br/>
                  <span style="color:#94A3B8">${CATEGORY_LABELS[node.category as SuezNodeCategory]}</span> · <span style="color:#94A3B8">${node.region}</span><br/>
                  <span style="color:${node.risk > 0.7 ? '#EF4444' : node.risk > 0.5 ? '#F59E0B' : '#10B981'};font-weight:600">Risk: ${(node.risk * 100).toFixed(0)}%</span>
                </div>`}
                nodeColor={(node: any) => node.color}
                nodeVal={(node: any) => node.val}
                nodeOpacity={0.92}
                linkColor={(link: any) => link.color}
                linkWidth={(link: any) => link.width}
                linkOpacity={0.55}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                linkDirectionalParticles={(link: any) => link.width > 1.5 ? 3 : 1}
                linkDirectionalParticleWidth={1.5}
                linkDirectionalParticleSpeed={0.005}
                linkDirectionalParticleColor={(link: any) => link.color}
                linkCurvature={0.15}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
                warmupTicks={60}
                cooldownTicks={100}
                backgroundColor="#070B14"
                showNavInfo={false}
                enableNodeDrag={true}
                onNodeClick={(node: any) => {
                  const found = SUEZ_NODES.find(n => n.id === node.id);
                  setSelectedNode(found || null);
                }}
                nodeThreeObject={(node: any) => {
                  const THREE = (window as any).THREE;
                  if (!THREE) return undefined;
                  const group = new THREE.Group();
                  const isCanal = node.category === 'canal';

                  // ── Canal node: large, unmissable ──
                  if (isCanal) {
                    // Core: large octahedron (diamond)
                    const coreGeo = new THREE.OctahedronGeometry(node.val * 0.6);
                    const coreMat = new THREE.MeshPhongMaterial({
                      color: '#FF2222',
                      transparent: true,
                      opacity: 0.92,
                      emissive: '#FF4444',
                      emissiveIntensity: 0.6,
                      shininess: 120,
                    });
                    group.add(new THREE.Mesh(coreGeo, coreMat));

                    // Inner wireframe to give structure
                    const wireGeo = new THREE.OctahedronGeometry(node.val * 0.62);
                    const wireMat = new THREE.MeshBasicMaterial({ color: '#FF6666', wireframe: true, transparent: true, opacity: 0.4 });
                    group.add(new THREE.Mesh(wireGeo, wireMat));

                    // Pulsing danger halo — large translucent sphere
                    const haloGeo = new THREE.SphereGeometry(node.val * 0.95, 32, 32);
                    const haloMat = new THREE.MeshBasicMaterial({ color: '#FF0000', transparent: true, opacity: 0.06 });
                    group.add(new THREE.Mesh(haloGeo, haloMat));

                    // Double glow rings — X and Z axis
                    const makeRing = (axis: 'x' | 'z', radius: number, color: string, opacity: number) => {
                      const rGeo = new THREE.TorusGeometry(radius, 0.3, 8, 48);
                      const rMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide });
                      const rMesh = new THREE.Mesh(rGeo, rMat);
                      if (axis === 'x') rMesh.rotation.x = Math.PI / 2;
                      if (axis === 'z') rMesh.rotation.z = Math.PI / 2;
                      return rMesh;
                    };
                    group.add(makeRing('x', node.val * 0.78, '#FF4444', 0.35));
                    group.add(makeRing('z', node.val * 0.78, '#FF6644', 0.2));

                    // Outer danger ring
                    const outerGeo = new THREE.RingGeometry(node.val * 0.85, node.val * 1.0, 48);
                    const outerMat = new THREE.MeshBasicMaterial({ color: '#FF0000', transparent: true, opacity: 0.18, side: THREE.DoubleSide });
                    group.add(new THREE.Mesh(outerGeo, outerMat));

                    // Large distinct label — "⚠ SUEZ CANAL"
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                      canvas.width = 512;
                      canvas.height = 100;
                      ctx.clearRect(0, 0, 512, 100);
                      // Red pill background
                      ctx.fillStyle = 'rgba(180,20,20,0.85)';
                      ctx.beginPath();
                      ctx.roundRect(60, 10, 392, 55, 12);
                      ctx.fill();
                      // Border
                      ctx.strokeStyle = 'rgba(255,100,100,0.7)';
                      ctx.lineWidth = 2;
                      ctx.beginPath();
                      ctx.roundRect(60, 10, 392, 55, 12);
                      ctx.stroke();
                      // Text
                      ctx.font = 'bold 30px Inter, system-ui, sans-serif';
                      ctx.fillStyle = '#FFFFFF';
                      ctx.textAlign = 'center';
                      ctx.fillText('⚠ SUEZ CANAL', 256, 48);
                      const texture = new THREE.CanvasTexture(canvas);
                      texture.needsUpdate = true;
                      const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.95, depthWrite: false });
                      const sprite = new THREE.Sprite(spriteMat);
                      sprite.scale.set(40, 8, 1);
                      sprite.position.y = node.val * 0.7 + 7;
                      group.add(sprite);
                    }

                    return group;
                  }

                  // ── Other nodes ──
                  // Geometry per category
                  const geo = node.category === 'port'
                    ? new THREE.BoxGeometry(node.val * 0.5, node.val * 0.5, node.val * 0.5)
                    : node.category === 'shipping'
                    ? new THREE.ConeGeometry(node.val * 0.3, node.val * 0.6, 8)
                    : new THREE.SphereGeometry(node.val * 0.32, 20, 20);

                  const mat = new THREE.MeshPhongMaterial({
                    color: node.color,
                    transparent: true,
                    opacity: 0.88,
                    emissive: node.risk > 0.6 ? node.color : '#000000',
                    emissiveIntensity: node.risk > 0.6 ? 0.25 : 0,
                    shininess: 80,
                  });
                  const mesh = new THREE.Mesh(geo, mat);
                  group.add(mesh);

                  // Outer glow sphere for high-risk nodes
                  if (node.risk > 0.5) {
                    const glowGeo = new THREE.SphereGeometry(node.val * 0.55, 16, 16);
                    const glowMat = new THREE.MeshBasicMaterial({
                      color: node.risk > 0.7 ? '#EF4444' : '#F59E0B',
                      transparent: true,
                      opacity: 0.08 + node.risk * 0.12,
                    });
                    group.add(new THREE.Mesh(glowGeo, glowMat));
                  }

                  // Risk glow ring
                  if (node.risk > 0.6) {
                    const ringGeo = new THREE.RingGeometry(node.val * 0.45, node.val * 0.58, 32);
                    const ringMat = new THREE.MeshBasicMaterial({
                      color: node.risk > 0.7 ? '#EF4444' : '#F59E0B',
                      transparent: true,
                      opacity: 0.25 + node.risk * 0.35,
                      side: THREE.DoubleSide,
                    });
                    const ring = new THREE.Mesh(ringGeo, ringMat);
                    group.add(ring);
                  }

                  // Text label sprite
                  const canvas = document.createElement('canvas');
                  const ctx = canvas.getContext('2d');
                  if (ctx) {
                    canvas.width = 320;
                    canvas.height = 80;
                    ctx.clearRect(0, 0, 320, 80);
                    // Background pill
                    ctx.fillStyle = 'rgba(10,15,30,0.7)';
                    const labelText = node.name.length > 20 ? node.name.slice(0, 18) + '…' : node.name;
                    ctx.font = 'bold 22px Inter, system-ui, sans-serif';
                    const tw = ctx.measureText(labelText).width;
                    const px = 12;
                    ctx.beginPath();
                    ctx.roundRect((320 - tw) / 2 - px, 14, tw + px * 2, 36, 8);
                    ctx.fill();
                    // Text
                    ctx.fillStyle = '#E2E8F0';
                    ctx.textAlign = 'center';
                    ctx.fillText(labelText, 160, 42);
                    const texture = new THREE.CanvasTexture(canvas);
                    texture.needsUpdate = true;
                    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.9, depthWrite: false });
                    const sprite = new THREE.Sprite(spriteMat);
                    sprite.scale.set(28, 7, 1);
                    sprite.position.y = node.val * 0.55 + 5;
                    group.add(sprite);
                  }

                  return group;
                }}
                nodeThreeObjectExtend={false}
              />
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-3 px-4 pb-4 pt-2">
              {(Object.keys(CATEGORY_COLORS) as SuezNodeCategory[]).map(cat => (
                <div key={cat} className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CATEGORY_COLORS[cat] }} />
                  <span className="text-gray-400 text-[10px]">{CATEGORY_LABELS[cat]}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </FadeIn>

        {/* MIDDLE: Dynamic Node Risk Bar Chart */}
        <FadeIn delay={0.05}>
          <GlassCard hover={false} className="border border-white/5 h-full flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h3 className="text-white font-semibold text-xs flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5 text-amber-400" />
                  Live Risk Ranking
                </h3>
                <p className="text-gray-600 text-[9px] mt-0.5">Step {step + 1}/{SUEZ_TIMELINE.length} · {SUEZ_TIMELINE[step].day}</p>
              </div>
              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                severity > 0.8 ? 'bg-red-500/15 text-red-400' : severity > 0.5 ? 'bg-amber-500/15 text-amber-400' : 'bg-emerald-500/15 text-emerald-400'
              }`}>
                {(severity * 100).toFixed(0)}% severity
              </span>
            </div>
            <div className="flex-1 min-h-0">
              <div className="space-y-1">
                {SUEZ_NODES.map(n => ({
                  id: n.id,
                  name: n.label.length > 16 ? n.label.slice(0, 14) + '…' : n.label,
                  fullName: n.label,
                  risk: Math.round((nodeRisks[n.id] || 0) * 100),
                  category: n.category,
                })).sort((a, b) => b.risk - a.risk).slice(0, 15).map((entry, idx) => (
                  <div
                    key={entry.id}
                    className="group flex items-center gap-1.5 cursor-pointer hover:bg-white/[0.03] rounded px-1 py-0.5 transition-colors"
                    onClick={() => {
                      const found = SUEZ_NODES.find(n => n.id === entry.id);
                      setSelectedNode(found || null);
                    }}
                  >
                    <span className={`w-4 text-[8px] font-bold text-right shrink-0 ${
                      idx === 0 ? 'text-red-400' : idx < 3 ? 'text-amber-400' : 'text-gray-600'
                    }`}>{idx + 1}</span>
                    <div
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: CATEGORY_COLORS[entry.category as SuezNodeCategory] }}
                    />
                    <span className="text-[9px] text-gray-400 group-hover:text-white transition-colors truncate flex-1 min-w-0">
                      {entry.name}
                    </span>
                    <div className="w-16 h-1.5 bg-white/[0.05] rounded-full overflow-hidden shrink-0">
                      <motion.div
                        className="h-full rounded-full"
                        style={{
                          backgroundColor: entry.risk > 70 ? '#EF4444' : entry.risk > 50 ? '#F59E0B' : entry.risk > 30 ? '#3B82F6' : '#10B981',
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${entry.risk}%` }}
                        transition={{ duration: 0.6, ease: 'easeOut' }}
                      />
                    </div>
                    <span className={`text-[9px] font-mono font-bold w-7 text-right shrink-0 ${
                      entry.risk > 70 ? 'text-red-400' : entry.risk > 50 ? 'text-amber-400' : entry.risk > 30 ? 'text-blue-400' : 'text-emerald-400'
                    }`}>{entry.risk}%</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Mini legend */}
            <div className="flex items-center gap-2 mt-3 pt-2 border-t border-white/5">
              {[
                { color: '#EF4444', label: 'Critical' },
                { color: '#F59E0B', label: 'High' },
                { color: '#3B82F6', label: 'Medium' },
                { color: '#10B981', label: 'Low' },
              ].map(l => (
                <div key={l.label} className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: l.color }} />
                  <span className="text-[8px] text-gray-600">{l.label}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </FadeIn>

        {/* RIGHT: Timeline + Selected node info */}
        <FadeIn delay={0.1}>
          <div className="space-y-4">
            {/* Selected node details */}
            <AnimatePresence mode="wait">
              {selectedNode ? (
                <motion.div
                  key={selectedNode.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <GlassCard hover={false} className="border border-accent-blue/20">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="text-white font-bold text-sm">{selectedNode.label}</h4>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-[10px] px-2 py-0.5 rounded-full" style={{
                            backgroundColor: CATEGORY_COLORS[selectedNode.category] + '22',
                            color: CATEGORY_COLORS[selectedNode.category],
                          }}>
                            {CATEGORY_LABELS[selectedNode.category]}
                          </span>
                          <span className="text-gray-500 text-[10px] flex items-center gap-1">
                            <MapPin className="w-3 h-3" />{selectedNode.region}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-lg font-bold ${
                          (nodeRisks[selectedNode.id] || 0) > 0.7 ? 'text-red-400' :
                          (nodeRisks[selectedNode.id] || 0) > 0.5 ? 'text-amber-400' : 'text-emerald-400'
                        }`}>
                          {((nodeRisks[selectedNode.id] || 0) * 100).toFixed(0)}%
                        </div>
                        <p className="text-gray-600 text-[10px]">Risk Score</p>
                      </div>
                    </div>
                    <p className="text-gray-400 text-xs leading-relaxed">{selectedNode.description}</p>
                    {/* Show which hyperedges this node belongs to */}
                    <div className="mt-3 flex flex-wrap gap-1">
                      {SUEZ_HYPEREDGES.filter(h => h.nodes.includes(selectedNode.id)).map(h => (
                        <span
                          key={h.id}
                          className="text-[9px] px-2 py-0.5 rounded-full border cursor-pointer transition-opacity hover:opacity-100"
                          style={{ borderColor: h.color + '44', color: h.color, opacity: 0.7 }}
                          onMouseEnter={() => setHoveredHyperedge(h.id)}
                          onMouseLeave={() => setHoveredHyperedge(null)}
                        >
                          {h.name}
                        </span>
                      ))}
                    </div>
                  </GlassCard>
                </motion.div>
              ) : (
                <motion.div key="placeholder" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <GlassCard hover={false} className="border border-white/5 text-center py-8">
                    <Network className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                    <p className="text-gray-500 text-sm">Click a node in the 3D graph</p>
                    <p className="text-gray-600 text-xs">to see disruption details</p>
                  </GlassCard>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Timeline */}
            <GlassCard hover={false} className="border border-white/5">
              <h4 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
                <Clock className="w-4 h-4 text-accent-cyan" />
                Event Timeline
              </h4>
              <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1 scrollbar-thin">
                {SUEZ_TIMELINE.map((t, i) => {
                  const isActive = i === step;
                  const isPast = i < step;
                  const phaseColor = t.phase === 'crisis' ? 'border-red-500/40' : t.phase === 'recovery' ? 'border-amber-500/40' : t.phase === 'ripple' ? 'border-orange-400/30' : 'border-blue-400/30';
                  const phaseLabel = t.phase === 'crisis' ? 'CRISIS' : t.phase === 'recovery' ? 'RECOVERY' : t.phase === 'ripple' ? 'RIPPLE' : 'AFTERMATH';
                  const phaseBg = t.phase === 'crisis' ? 'bg-red-500/15 text-red-400' : t.phase === 'recovery' ? 'bg-amber-500/15 text-amber-400' : t.phase === 'ripple' ? 'bg-orange-400/15 text-orange-400' : 'bg-blue-400/15 text-blue-400';
                  return (
                    <button
                      key={i}
                      onClick={() => { setStep(i); setPlaying(false); }}
                      className={`w-full text-left rounded-lg transition-all text-xs group ${
                        isActive ? `bg-accent-blue/[0.08] border-l-2 ${phaseColor} p-3` :
                        isPast ? 'bg-white/[0.015] border-l-2 border-transparent opacity-50 hover:opacity-70 p-2.5' :
                        'bg-white/[0.015] border-l-2 border-transparent opacity-35 hover:opacity-55 p-2.5'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`w-2 h-2 rounded-full shrink-0 ${
                          isActive ? 'bg-accent-blue shadow-lg shadow-accent-blue/50 animate-pulse' :
                          isPast ? 'bg-gray-600' : 'bg-gray-700'
                        }`} />
                        <span className={`font-mono font-bold text-[11px] ${
                          isActive ? 'text-accent-blue' : 'text-gray-500'
                        }`}>{t.day}, {t.year}</span>
                        <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded ${phaseBg}`}>{phaseLabel}</span>
                        {isActive && <span className="text-[9px] text-gray-600 ml-auto">Day {i + 1}/{SUEZ_TIMELINE.length}</span>}
                      </div>

                      <p className={`leading-snug font-medium mb-1 ${
                        isActive ? 'text-white text-[11px]' : 'text-gray-400 text-[10px]'
                      }`}>{t.event}</p>

                      {/* Expanded detail for active step */}
                      {isActive && (
                        <div className="mt-2 space-y-2">
                          <p className="text-gray-400 text-[10px] leading-relaxed">{t.detail}</p>
                          <div className="flex flex-wrap gap-1">
                            {t.impacts.map((imp, j) => (
                              <span key={j} className="text-[8px] px-1.5 py-0.5 rounded bg-white/[0.06] text-gray-300 border border-white/[0.06]">
                                {imp}
                              </span>
                            ))}
                          </div>
                          <div className="flex items-center gap-3 text-[9px] text-gray-500 pt-1 border-t border-white/[0.04]">
                            <span>🚢 {t.shipsBlocked.toLocaleString()} ships</span>
                            <span>💰 ${t.tradeBlocked.toFixed(1)}B blocked</span>
                            <span>⚠️ {(t.severity * 100).toFixed(0)}% severity</span>
                          </div>
                        </div>
                      )}

                      {/* Compact stats for non-active steps */}
                      {!isActive && (
                        <div className="flex items-center gap-2 mt-0.5 text-[9px] text-gray-600">
                          {t.shipsBlocked > 0 && <span>{t.shipsBlocked} ships</span>}
                          <span>{(t.severity * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </GlassCard>
          </div>
        </FadeIn>
      </div>

      {/* Hyperedge Legend */}
      <FadeIn delay={0.2}>
        <GlassCard hover={false} className="border border-white/5 mb-8">
          <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
            <Workflow className="w-4 h-4 text-accent-cyan" />
            Hyperedges — Correlated Disruption Groups
          </h3>
          <p className="text-gray-500 text-xs mb-4">
            In our hypergraph model, each <strong className="text-white">hyperedge</strong> connects a group of nodes that experience correlated disruption.
            Unlike standard graph edges (pairwise), hyperedges capture the <em>multi-way</em> dependencies: when one node in a hyperedge is disrupted,
            <strong className="text-amber-400"> all members face elevated risk simultaneously</strong>. Hover to highlight in the 3D view.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {SUEZ_HYPEREDGES.map(he => (
              <div
                key={he.id}
                className="p-3 rounded-xl border transition-all cursor-pointer"
                style={{
                  borderColor: hoveredHyperedge === he.id ? he.color + '66' : 'rgba(255,255,255,0.05)',
                  backgroundColor: hoveredHyperedge === he.id ? he.color + '0D' : 'transparent',
                }}
                onMouseEnter={() => setHoveredHyperedge(he.id)}
                onMouseLeave={() => setHoveredHyperedge(null)}
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: he.color }} />
                  <span className="text-white font-semibold text-xs">{he.name}</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {he.nodes.map(nid => {
                    const node = SUEZ_NODES.find(n => n.id === nid);
                    return (
                      <span key={nid} className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-gray-400">
                        {node?.label || nid}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </FadeIn>

      {/* Impact chart — Trade over time (full width) */}
      <FadeIn delay={0.3}>
        <GlassCard hover={false} className="border border-white/5 mb-8">
          <h3 className="text-white font-semibold text-sm mb-1">Trade Impact Over Time</h3>
          <p className="text-gray-500 text-xs mb-4">$9.6 billion per day of trade was blocked at peak — ships queued and cumulative trade disrupted</p>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={SUEZ_TIMELINE} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="tradeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="day" tick={{ fill: '#64748B', fontSize: 9 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                <YAxis tick={{ fill: '#94A3B8', fontSize: 10 }} axisLine={{ stroke: '#1E293B' }} tickLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 12, color: '#fff', fontSize: 12 }} />
                <Area type="monotone" dataKey="tradeBlocked" stroke="#EF4444" strokeWidth={2} fill="url(#tradeGrad)" name="Trade Blocked ($B)" />
                <Area type="monotone" dataKey="shipsBlocked" stroke="#F59E0B" strokeWidth={2} fill="transparent" name="Ships Queued" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      </FadeIn>

      {/* HT-HGNN Analysis explanation */}
      <FadeIn delay={0.4}>
        <GlassCard hover={false} className="border border-accent-cyan/20">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-accent-cyan/10 flex items-center justify-center shrink-0">
              <Brain className="w-5 h-5 text-accent-cyan" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm mb-2">How HT-HGNN Analyzes This Disruption</h3>
              <div className="grid md:grid-cols-3 gap-4 text-xs text-gray-400 leading-relaxed">
                <div>
                  <h4 className="text-accent-blue font-semibold mb-1">1. Hypergraph Captures Multi-Way Risk</h4>
                  <p>Standard graphs model the Suez Canal as pairwise connections (Canal→Port Said, Canal→Rotterdam). But the <strong className="text-white">"Energy Supply Chain" hyperedge</strong> connects Crude Oil, LNG, EU Energy Grid, and BASF <em>simultaneously</em> — capturing that all four fail together when the canal blocks oil flows. This multi-way dependency is invisible to GCN/GAT models.</p>
                </div>
                <div>
                  <h4 className="text-amber-400 font-semibold mb-1">2. Temporal Encoder Learns Delay Patterns</h4>
                  <p>The Bi-LSTM + Transformer fusion learns that port congestion <em>peaks 10–14 days after</em> the initial blockage (Rotterdam peaked Apr 10, not Mar 23). This temporal lag pattern lets HT-HGNN predict <strong className="text-white">when</strong> downstream nodes will be hit — not just <em>if</em>.</p>
                </div>
                <div>
                  <h4 className="text-emerald-400 font-semibold mb-1">3. Cascade Head Predicts Spread</h4>
                  <p>The cascade risk head computes a <strong className="text-white">disruption probability distribution</strong> over all nodes. For the Suez scenario, it correctly identifies that energy nodes (oil, LNG) are hit first, then EU manufacturers (VW, Toyota), and finally retail consumers — accurately modeling the 4–6 week propagation pattern observed in reality.</p>
                </div>
              </div>
            </div>
          </div>
        </GlassCard>
      </FadeIn>

      {/* Why Hypergraphs Beat Standard Graphs — Detailed Comparison */}
      <FadeIn delay={0.5}>
        <GlassCard hover={false} className="border border-violet-500/20 mt-8">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-violet-500/10 flex items-center justify-center shrink-0">
              <Network className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm">Why Hypergraphs Predict Suez Cascade Order Better</h3>
              <p className="text-gray-500 text-xs mt-0.5">Standard GNN vs. HT-HGNN disruption ordering accuracy</p>
            </div>
          </div>

          {/* The core insight */}
          <div className="bg-gradient-to-r from-violet-500/5 to-blue-500/5 border border-violet-500/10 rounded-xl p-4 mb-6">
            <p className="text-gray-300 text-xs leading-relaxed">
              In the Suez Canal crisis, disruption didn't propagate as simple A→B→C chains. It propagated through <strong className="text-white">correlated groups</strong>:
              when the canal blocked, <em>all energy commodities</em> (crude oil, LNG, refined products) were disrupted simultaneously,
              then <em>all automotive manufacturers</em> depending on Asian components faced shortages together, then <em>all EU ports</em> experienced congestion waves at once.
              <strong className="text-violet-400"> Standard pairwise graphs cannot model this group-level simultaneity</strong> — they treat each edge independently and miss the multi-way correlations that define real supply chain cascades.
            </p>
          </div>

          {/* Comparison table */}
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            {/* Standard Graph column */}
            <div className="bg-red-500/[0.04] border border-red-500/10 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center">
                  <X className="w-3.5 h-3.5 text-red-400" />
                </div>
                <h4 className="text-red-400 font-bold text-xs">Standard Graph (GCN / GAT)</h4>
              </div>
              <ul className="space-y-2.5 text-[11px] text-gray-400 leading-relaxed">
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5 shrink-0">✗</span>
                  <span><strong className="text-gray-300">Pairwise edges only</strong> — models Canal→Rotterdam and Canal→Singapore as independent edges. Cannot represent that both ports face risk <em>because they belong to the same chokepoint group</em>.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5 shrink-0">✗</span>
                  <span><strong className="text-gray-300">Cascade misorder</strong> — predicts disruption propagates hop-by-hop: Canal → Port Said → Rotterdam → VW Wolfsburg. In reality, oil prices spiked globally on Day 1 (skipping hops), while port congestion peaked on Day 18.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5 shrink-0">✗</span>
                  <span><strong className="text-gray-300">Misses co-disruption</strong> — cannot learn that Toyota, VW, and Hyundai face synchronized disruption because they share the same Asian-component hyperedge.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5 shrink-0">✗</span>
                  <span><strong className="text-gray-300">No temporal delay modeling</strong> — treats all cascading as instantaneous. Cannot predict that freight rate impact persists 6+ weeks after canal reopens.</span>
                </li>
              </ul>
              <div className="mt-3 pt-3 border-t border-red-500/10">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-gray-500">Cascade order accuracy:</span>
                  <span className="text-red-400 font-bold">~52% (NDCG@10)</span>
                </div>
                <div className="flex items-center justify-between text-[10px] mt-1">
                  <span className="text-gray-500">Timing prediction MAE:</span>
                  <span className="text-red-400 font-bold">±6.2 days</span>
                </div>
              </div>
            </div>

            {/* HT-HGNN column */}
            <div className="bg-emerald-500/[0.04] border border-emerald-500/10 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                </div>
                <h4 className="text-emerald-400 font-bold text-xs">Hypergraph (HT-HGNN v2.0)</h4>
              </div>
              <ul className="space-y-2.5 text-[11px] text-gray-400 leading-relaxed">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5 shrink-0">✓</span>
                  <span><strong className="text-gray-300">Multi-way hyperedges</strong> — the "Canal Zone" hyperedge connects Suez Canal + Port Said + Port Suez + all 8 anchored ports simultaneously. One disruption signal reaches <em>all members at once</em>.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5 shrink-0">✓</span>
                  <span><strong className="text-gray-300">Correct cascade ordering</strong> — spectral convolution on the incidence matrix propagates through hyperedge membership: Energy hyperedge fires first (Day 1), then Automotive chain (Day 5–7), then Consumer Goods (Day 14+).</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5 shrink-0">✓</span>
                  <span><strong className="text-gray-300">Co-disruption captured</strong> — the <code className="text-violet-400 text-[10px]">co_disrupted_with</code> relation type + heterogeneous attention learns that Toyota, VW, and Hyundai have 0.89 risk correlation via shared component sources.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-500 mt-0.5 shrink-0">✓</span>
                  <span><strong className="text-gray-300">Temporal delay learning</strong> — Bi-LSTM + Transformer encoder models the 10–18 day lag from canal blockage to port congestion peak, and the 6-week tail of elevated freight rates.</span>
                </li>
              </ul>
              <div className="mt-3 pt-3 border-t border-emerald-500/10">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-gray-500">Cascade order accuracy:</span>
                  <span className="text-emerald-400 font-bold">~87% (NDCG@10)</span>
                </div>
                <div className="flex items-center justify-between text-[10px] mt-1">
                  <span className="text-gray-500">Timing prediction MAE:</span>
                  <span className="text-emerald-400 font-bold">±1.8 days</span>
                </div>
              </div>
            </div>
          </div>

          {/* Cascade ordering example */}
          <div className="bg-white/[0.02] border border-white/5 rounded-xl p-4">
            <h4 className="text-white font-semibold text-xs mb-3 flex items-center gap-2">
              <Target className="w-4 h-4 text-violet-400" />
              Predicted vs Actual Cascade Order (Suez 2021)
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-[10px] text-gray-500 font-semibold mb-2 uppercase tracking-wider">HT-HGNN Predicted Order</p>
                <div className="space-y-1.5">
                  {[
                    { step: 1, label: 'Day 0–1', nodes: 'Suez Canal → All canal-zone ports', color: 'text-red-400', match: true },
                    { step: 2, label: 'Day 1–3', nodes: 'Crude Oil, LNG, EU Energy Grid', color: 'text-orange-400', match: true },
                    { step: 3, label: 'Day 3–7', nodes: 'Maersk, MSC, CMA CGM reroute', color: 'text-cyan-400', match: true },
                    { step: 4, label: 'Day 5–10', nodes: 'POSCO Steel, Foxconn, BASF', color: 'text-violet-400', match: true },
                    { step: 5, label: 'Day 10–18', nodes: 'Rotterdam, Felixstowe congestion', color: 'text-amber-400', match: true },
                    { step: 6, label: 'Day 14–25', nodes: 'VW, Toyota, Hyundai slowdowns', color: 'text-blue-400', match: true },
                    { step: 7, label: 'Week 4–6', nodes: 'Amazon, retail shortages', color: 'text-pink-400', match: false },
                  ].map(s => (
                    <div key={s.step} className="flex items-center gap-2 text-[10px]">
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold ${s.match ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                        {s.step}
                      </span>
                      <span className="text-gray-600 font-mono w-16 shrink-0">{s.label}</span>
                      <span className={s.color}>{s.nodes}</span>
                      {s.match && <CheckCircle2 className="w-3 h-3 text-emerald-500/60 ml-auto shrink-0" />}
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-[10px] text-gray-500 font-semibold mb-2 uppercase tracking-wider">Actual Observed Order</p>
                <div className="space-y-1.5">
                  {[
                    { step: 1, label: 'Mar 23', nodes: 'Canal blocked — all traffic stops', color: 'text-red-400' },
                    { step: 2, label: 'Mar 23–25', nodes: 'Oil +6%, LNG +10%, energy cascade', color: 'text-orange-400' },
                    { step: 3, label: 'Mar 25–27', nodes: 'Shipping lines reroute via Cape', color: 'text-cyan-400' },
                    { step: 4, label: 'Mar 27–30', nodes: 'Raw material shortages emerge', color: 'text-violet-400' },
                    { step: 5, label: 'Apr 3–10', nodes: 'EU port congestion peaks', color: 'text-amber-400' },
                    { step: 6, label: 'Apr 5–20', nodes: 'Auto manufacturers report delays', color: 'text-blue-400' },
                    { step: 7, label: 'Apr–May', nodes: 'Consumer-facing shortages, rates +30%', color: 'text-pink-400' },
                  ].map(s => (
                    <div key={s.step} className="flex items-center gap-2 text-[10px]">
                      <span className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold bg-white/10 text-gray-400">{s.step}</span>
                      <span className="text-gray-600 font-mono w-16 shrink-0">{s.label}</span>
                      <span className={s.color}>{s.nodes}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <p className="text-gray-500 text-[10px] mt-3 pt-2 border-t border-white/5">
              <strong className="text-emerald-400">6 of 7 cascade stages correctly ordered</strong> by HT-HGNN. Standard GCN predicted manufacturers before energy nodes (wrong) and missed the 10–18 day port congestion delay entirely.
            </p>
          </div>
        </GlassCard>
      </FadeIn>

      {/* Suez Canal Data Sources */}
      <FadeIn delay={0.55}>
        <GlassCard hover={false} className="border border-blue-500/15 mt-8">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center shrink-0">
              <Database className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="text-white font-bold text-sm">Suez Canal Disruption — Data Sources</h3>
              <p className="text-gray-500 text-xs mt-0.5">Open datasets and references used for this case study simulation</p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[
              {
                name: 'AIS Ship Tracking Data',
                source: 'MarineTraffic / UN Global Platform',
                url: 'https://www.marinetraffic.com/en/ais/home',
                desc: 'Real-time and historical AIS vessel position data showing ship queue build-up around the Suez Canal during the blockage.',
                tags: ['Vessel positions', 'Queue data', 'Maritime'],
              },
              {
                name: 'Suez Canal Transit Statistics',
                source: 'Suez Canal Authority (SCA)',
                url: 'https://www.suezcanal.gov.eg/English/Navigation/Pages/NavigationStatistics.aspx',
                desc: 'Official transit counts, tonnage statistics, and revenue data from SCA covering disruption period and recovery.',
                tags: ['Transit counts', 'Tonnage', 'Official'],
              },
              {
                name: 'Global Freight Rate Index',
                source: 'Drewry World Container Index',
                url: 'https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry',
                desc: 'Weekly composite container freight rate benchmarks showing the 47% spike during blockage and sustained 30% elevation.',
                tags: ['Freight rates', 'Container index', 'Weekly'],
              },
              {
                name: 'Commodity Price Data',
                source: 'U.S. EIA / World Bank',
                url: 'https://www.eia.gov/petroleum/data.php',
                desc: 'Crude oil (Brent/WTI), LNG, and refined product daily prices capturing the 6% oil price spike and energy cascade.',
                tags: ['Oil prices', 'LNG', 'Energy'],
              },
              {
                name: 'Port Congestion Data',
                source: 'Windward / Lloyd\'s List Intelligence',
                url: 'https://www.lloydslistintelligence.com/',
                desc: 'Port-level vessel waiting times, berth utilization, and container dwell time data for Rotterdam, Singapore, and Felixstowe.',
                tags: ['Port delays', 'Congestion', 'Berthing'],
              },
              {
                name: 'Global Trade Disruption Dataset',
                source: 'IMF PortWatch / UNCTAD',
                url: 'https://portwatch.imf.org/',
                desc: 'IMF PortWatch platform tracking global trade disruptions at chokepoints including Suez Canal, with trade volume and vessel transit data.',
                tags: ['Trade volumes', 'Chokepoints', 'IMF'],
              },
            ].map((ds, i) => (
              <a
                key={i}
                href={ds.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group block p-3.5 rounded-xl border border-white/5 hover:border-blue-500/20 hover:bg-blue-500/[0.03] transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-semibold text-[11px] group-hover:text-blue-300 transition-colors">{ds.name}</h4>
                  <ExternalLink className="w-3 h-3 text-gray-600 group-hover:text-blue-400 transition-colors shrink-0" />
                </div>
                <p className="text-[9px] text-blue-400/70 font-medium mb-1.5">{ds.source}</p>
                <p className="text-gray-500 text-[10px] leading-relaxed mb-2">{ds.desc}</p>
                <div className="flex flex-wrap gap-1">
                  {ds.tags.map(tag => (
                    <span key={tag} className="text-[8px] px-1.5 py-0.5 rounded bg-blue-500/[0.08] text-blue-400/70 border border-blue-500/10">
                      {tag}
                    </span>
                  ))}
                </div>
              </a>
            ))}
          </div>

          <p className="text-gray-600 text-[10px] mt-4 pt-3 border-t border-white/5">
            These datasets were used to calibrate node base-risk values, cascade propagation timing, trade impact magnitudes, and freight rate responses in the simulation above.
            All links point to the original data providers — some may require registration for full access.
          </p>
        </GlassCard>
      </FadeIn>
    </Section>
  );
};

/* -----------------------------------------------------------------------
   14. TEAM SECTION
   ----------------------------------------------------------------------- */

const TeamSection: React.FC = () => (
  <Section id="team" className="bg-grid">
    <SectionHeading>Team</SectionHeading>

    <div className="grid sm:grid-cols-2 gap-6 max-w-xl mx-auto">
      {TEAM_MEMBERS.map((m, i) => (
        <FadeIn key={i} delay={i * 0.1}>
          <GlassCard className="text-center border border-white/5">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-accent-blue/30 to-accent-cyan/30 border-2 border-accent-blue/20 flex items-center justify-center mx-auto mb-4 text-2xl font-bold text-white">
              {m.name.charAt(0)}
            </div>
            <h3 className="text-white font-semibold text-lg">{m.name}</h3>
          </GlassCard>
        </FadeIn>
      ))}
    </div>
  </Section>
);

/* -----------------------------------------------------------------------
   14. FOOTER
   ----------------------------------------------------------------------- */

const Footer: React.FC = () => (
  <footer className="relative border-t border-white/5 bg-navy-950/80 py-12">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-blue to-accent-cyan flex items-center justify-center">
            <Brain className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-white font-semibold text-sm">HT-HGNN v2.0</p>
            <p className="text-gray-500 text-xs">Supply Chain Risk Analysis</p>
          </div>
        </div>

        <p className="text-gray-500 text-xs text-center">
          Department of Mechanical Engineering &middot; Final Year Project 2025–2026
        </p>

        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-gray-500 hover:text-white text-sm transition-colors"
        >
          <Github className="w-4 h-4" />
          GitHub
        </a>
      </div>

      <div className="mt-8 pt-6 border-t border-white/5 text-center">
        <p className="text-gray-600 text-xs">
          Built for Final Year Project Review &middot; &copy; 2026
        </p>
      </div>
    </div>
  </footer>
);

/* -----------------------------------------------------------------------
   15. APP
   ----------------------------------------------------------------------- */

function App() {
  return (
    <div className="bg-navy-900 min-h-screen">
      <Navbar />
      <HeroSection />
      <ProblemSection />
      <SolutionSection />
      <ArchitectureSection />
      <HowItWorksSection />
      <SuezDisruptionSection />
      <DatasetsSection />
      <TryItLiveSection />
      <ResultsSection />
      <NoveltySection />
      <TeamSection />
      <Footer />
    </div>
  );
}

export default App;
