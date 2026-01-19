import { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { RiskDashboard } from './components/RiskDashboard';
import { ModelStats } from './components/ModelStats';
import { Header } from './components/Header';
import { VisualizationPage } from './pages/VisualizationPage';
import inferenceService from './services/inferenceService';
import type { ModelInfo, AnalysisResult } from './types';

type PageType = 'home' | 'analysis' | 'model' | 'about' | 'visualization';

function App() {
  const [currentPage, setCurrentPage] = useState<PageType>('home');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch model info on mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await inferenceService.getModelInfo();
        setModelInfo(info);
      } catch (err) {
        console.error('Failed to fetch model info:', err);
        setError('Failed to load model information. Make sure the backend is running.');
      }
    };

    fetchModelInfo();
  }, []);

  const getRiskLevel = (score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' => {
    if (score < 0.25) return 'LOW';
    if (score < 0.5) return 'MEDIUM';
    if (score < 0.75) return 'HIGH';
    return 'CRITICAL';
  };

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await inferenceService.uploadAndPredict(file);
      
      // Convert predictions to risk assessments
      const riskAssessments = result.nodeIds.map((nodeId, idx) => ({
        nodeId,
        criticalityScore: result.criticalityScores[idx],
        riskLevel: getRiskLevel(result.criticalityScores[idx]),
        pricePrediction: result.pricePredictions[idx],
        changeForecasted: result.changePredictions[idx],
        recommendations: generateRecommendations(result.criticalityScores[idx]),
      }));

      // Sort by criticality and get top critical nodes
      const topCriticalNodes = [...riskAssessments]
        .sort((a, b) => b.criticalityScore - a.criticalityScore)
        .slice(0, 10);

      // Calculate average risk
      const averageRisk =
        riskAssessments.reduce((sum, ra) => sum + ra.criticalityScore, 0) /
        riskAssessments.length;

      setAnalysisResult({
        timestamp: new Date().toISOString(),
        predictions: result,
        riskAssessments,
        topCriticalNodes,
        averageRisk,
      });
      setCurrentPage('analysis');
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const generateRecommendations = (criticalityScore: number): string[] => {
    const recommendations: string[] = [];
    
    if (criticalityScore > 0.75) {
      recommendations.push('Implement backup suppliers immediately');
      recommendations.push('Increase safety stock levels');
      recommendations.push('Establish direct monitoring protocols');
    } else if (criticalityScore > 0.5) {
      recommendations.push('Evaluate alternative suppliers');
      recommendations.push('Review lead time forecasts');
      recommendations.push('Monitor market trends closely');
    } else if (criticalityScore > 0.25) {
      recommendations.push('Continue routine monitoring');
      recommendations.push('Update contingency plans');
    } else {
      recommendations.push('Maintain standard procedures');
      recommendations.push('Annual risk review sufficient');
    }
    
    return recommendations;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header currentPage={currentPage} onNavigate={setCurrentPage} />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        {currentPage === 'home' && (
          <div className="space-y-8">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg p-8">
              <h1 className="text-3xl font-bold mb-2">Supply Chain Risk Analysis</h1>
              <p className="text-blue-100">
                Powered by HT-HGNN (Heterogeneous Temporal Hypergraph Neural Network)
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Analyze Your Data</h2>
                <FileUpload onUpload={handleFileUpload} isLoading={isLoading} />
              </div>

              {modelInfo && (
                <div className="space-y-4">
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Model Information</h2>
                  <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-3">
                    <div>
                      <p className="text-sm text-gray-600">Name</p>
                      <p className="font-semibold text-gray-900">{modelInfo.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Parameters</p>
                      <p className="font-semibold text-gray-900">
                        {modelInfo.parameters.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Device</p>
                      <p className="font-semibold text-gray-900">{modelInfo.device}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Training Date</p>
                      <p className="font-semibold text-gray-900">
                        {new Date(modelInfo.trainingDate).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {currentPage === 'analysis' && analysisResult && (
          <RiskDashboard
            riskAssessments={analysisResult.riskAssessments}
            topCriticalNodes={analysisResult.topCriticalNodes}
            averageRisk={analysisResult.averageRisk}
          />
        )}

        {currentPage === 'visualization' && analysisResult && (
          <VisualizationPage riskAssessments={analysisResult.riskAssessments} />
        )}

        {currentPage === 'model' && modelInfo && (
          <ModelStats metrics={modelInfo.metrics} device={modelInfo.device} />
        )}

        {currentPage === 'about' && (
          <div className="max-w-3xl">
            <div className="bg-white border border-gray-200 rounded-lg p-8">
              <h1 className="text-2xl font-bold text-gray-900 mb-4">About HT-HGNN</h1>

              <div className="space-y-6 text-gray-700">
                <p>
                  The Heterogeneous Temporal Hypergraph Neural Network (HT-HGNN) is an advanced
                  machine learning model designed to analyze supply chain networks and assess risk
                  factors.
                </p>

                <div>
                  <h2 className="text-lg font-semibold text-gray-900 mb-2">Key Features</h2>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Multi-layer neural architecture (HGNN+ + HGT + TGN)</li>
                    <li>Real supply chain data analysis</li>
                    <li>Price prediction and change forecasting</li>
                    <li>Criticality scoring for supply chain nodes</li>
                    <li>GPU-optimized inference</li>
                  </ul>
                </div>

                <div>
                  <h2 className="text-lg font-semibold text-gray-900 mb-2">Architecture</h2>
                  <p>
                    The model combines three neural paradigms:
                  </p>
                  <ul className="list-disc list-inside space-y-2 mt-2">
                    <li>
                      <strong>HGNN+ (Hypergraph NN):</strong> Models multi-way relationships
                      between supply chain entities
                    </li>
                    <li>
                      <strong>HGT (Heterogeneous Graph Transformer):</strong> Handles different
                      node and edge types with attention mechanisms
                    </li>
                    <li>
                      <strong>TGN (Temporal Graph Network):</strong> Captures temporal dynamics
                      and cascade propagation
                    </li>
                  </ul>
                </div>

                <div>
                  <h2 className="text-lg font-semibold text-gray-900 mb-2">Use Cases</h2>
                  <ul className="list-disc list-inside space-y-2">
                    <li>Identify critical suppliers and vulnerabilities</li>
                    <li>Predict price changes and market impacts</li>
                    <li>Assess cascade risks in supply networks</li>
                    <li>Support strategic sourcing decisions</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-300 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          <p>HT-HGNN Supply Chain Risk Analysis Platform © 2026</p>
          <p className="mt-1">Powered by PyTorch and React</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
