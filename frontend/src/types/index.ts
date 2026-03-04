export interface PredictionInput {
  nodeFeatures: number[][];
  incidenceMatrix?: number[][];
}

export interface PredictionOutput {
  pricePredictions: number[];
  changePredictions: number[];
  criticalityScores: number[];
  nodeIds: string[];
}

export interface RiskAssessment {
  nodeId: string;
  criticalityScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  pricePrediction: number;
  changeForecasted: number;
  recommendations: string[];
}

export interface AnalysisResult {
  timestamp: string;
  predictions: PredictionOutput;
  riskAssessments: RiskAssessment[];
  topCriticalNodes: RiskAssessment[];
  averageRisk: number;
}

export interface TrainingMetrics {
  initialLoss: number;
  finalLoss: number;
  improvement: number;
  epochs: number;
  totalTime: number;
  memoryUsage: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  parameters: number;
  device: string;
  architecture: string;
  trainingDate: string;
  metrics: TrainingMetrics;
}

// ============================================================
// V2.0 Dataset Types
// ============================================================

/** Metadata describing an available dataset */
export interface DatasetInfo {
  id: string;
  name: string;
  nodes: number;
  hyperedges: number;
  timeSpan: string;
  primaryTask: string;
  source: string;
  description: string;
}

/** High-level summary statistics for a loaded graph */
export interface GraphSummary {
  nodes: number;
  hyperedges: number;
  avgHyperedgeSize: number;
  density: number;
}

/** Options when loading a dataset into memory */
export interface DatasetLoadRequest {
  temporalWindow?: number;
  minHyperedgeSize?: number;
  dynamicEdges?: boolean;
}

// ============================================================
// V2.0 Hypergraph Visualization Types
// ============================================================

/** A single node inside the hypergraph */
export interface HypergraphNode {
  id: string;
  label: string;
  type: string;
  riskLevel?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  riskScore?: number;
  features?: Record<string, number>;
  x?: number;
  y?: number;
}

/** A hyperedge connecting multiple nodes */
export interface HypergraphEdge {
  id: string;
  nodeIds: string[];
  type: string;
  weight: number;
  color?: string;
}

/** Full hypergraph payload returned by the API */
export interface HypergraphData {
  nodes: HypergraphNode[];
  hyperedges: HypergraphEdge[];
  metadata?: Record<string, any>;
}

// ============================================================
// V2.0 Cascade Simulation Types
// ============================================================

/** Parameters for triggering a cascade simulation */
export interface CascadeRequest {
  shockNodes: string[];
  shockMagnitude: number;
  propagationSteps: number;
  cascadeThreshold: number;
}

/** One discrete step in a cascade timeline */
export interface CascadeStep {
  step: number;
  disruptedCount: number;
  newlyDisrupted: string[];
  atRisk: string[];
}

/** Full result of a completed (or in-progress) cascade simulation */
export interface CascadeResult {
  taskId: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  timeline: CascadeStep[];
  totalDisrupted: number;
  criticalPaths: string[][];
  counterfactuals?: Record<string, number>;
}

// ============================================================
// V2.0 Explainability Types
// ============================================================

/** SHAP-based attribution for a single hyperedge */
export interface HyperedgeAttribution {
  edgeId: string;
  shapValue: number;
  type: string;
}

/** SHAP-based attribution for a single node feature */
export interface FeatureAttribution {
  feature: string;
  shapValue: number;
}

/** Full explanation payload for one node */
export interface NodeExplanation {
  nodeId: string;
  predictedRisk: string;
  confidence: number;
  hyperedgeAttributions: HyperedgeAttribution[];
  featureAttributions: FeatureAttribution[];
  recommendation: string;
}

/** Request body for the /explain endpoint */
export interface ExplainRequest {
  nodeIds: string[];
  predictionType: 'criticality' | 'price' | 'change';
  maxHyperedges?: number;
}

/** Response from the /explain endpoint */
export interface ExplainResponse {
  explanations: NodeExplanation[];
}

// ============================================================
// V2.0 Training Types
// ============================================================

/** Real-time streaming data emitted during model training */
export interface TrainingStreamData {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  metrics: Record<string, number>;
}
