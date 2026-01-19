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
