import { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { RiskDashboard } from './components/RiskDashboard';
import { ModelStats } from './components/ModelStats';
import { Header } from './components/Header';
import { VisualizationPage } from './pages/VisualizationPage';
import { SparklesCore } from './ui/sparkles';
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
    <div className="min-h-screen bg-black">
      {/* Main Sparkles Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 h-screen w-screen">
          <SparklesCore
            id="tsparticlesfullpage"
            background="transparent"
            minSize={0.6}
            maxSize={1.4}
            particleDensity={100}
            className="w-full h-full"
            particleColor="#FFFFFF"
          />
        </div>
      </div>

      {/* Content overlay */}
      <div className="relative z-10">
        <Header currentPage={currentPage} onNavigate={setCurrentPage} />

        <main className="min-h-screen">
          {error && (
            <div className="max-w-7xl mx-auto px-4 mt-8 mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}

          {currentPage === 'home' && (
            <div className="max-w-7xl mx-auto px-4 py-12">
              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-white mb-4">Analyze Your Data</h2>
                  <FileUpload onUpload={handleFileUpload} isLoading={isLoading} />
                </div>

                {modelInfo && (
                  <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg shadow-lg p-8 space-y-4">
                    <h2 className="text-2xl font-bold text-white mb-4">Model Information</h2>
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-gray-400">Name</p>
                        <p className="font-semibold text-white">{modelInfo.name}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Parameters</p>
                        <p className="font-semibold text-white">
                          {modelInfo.parameters.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Device</p>
                        <p className="font-semibold text-white">{modelInfo.device}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Training Date</p>
                        <p className="font-semibold text-white">
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
            <div className="max-w-7xl mx-auto px-4 py-8">
              <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg p-8">
                <RiskDashboard
                  riskAssessments={analysisResult.riskAssessments}
                  topCriticalNodes={analysisResult.topCriticalNodes}
                  averageRisk={analysisResult.averageRisk}
                />
              </div>
            </div>
          )}

          {currentPage === 'visualization' && analysisResult && (
            <div className="max-w-7xl mx-auto px-4 py-8">
              <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg p-8">
                <VisualizationPage riskAssessments={analysisResult.riskAssessments} />
              </div>
            </div>
          )}

          {currentPage === 'model' && modelInfo && (
            <div className="max-w-7xl mx-auto px-4 py-8">
              <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg p-8">
                <ModelStats metrics={modelInfo.metrics} device={modelInfo.device} />
              </div>
            </div>
          )}

          {currentPage === 'about' && (
            <div className="max-w-3xl mx-auto px-4 py-8">
              <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-lg p-8">
                <h1 className="text-2xl font-bold text-white mb-4">About HT-HGNN</h1>

                <div className="space-y-6 text-gray-300">
                  <p>
                    The Heterogeneous Temporal Hypergraph Neural Network (HT-HGNN) is an advanced
                    machine learning model designed to analyze supply chain networks and assess risk
                    factors.
                  </p>

                  <div>
                    <h2 className="text-lg font-semibold text-white mb-2">Key Features</h2>
                    <ul className="list-disc list-inside space-y-2">
                      <li>Multi-layer neural architecture</li>
                      <li>Real supply chain data analysis</li>
                      <li>Price prediction and change forecasting</li>
                      <li>Criticality scoring for supply chain nodes</li>
                      <li>GPU-optimized inference</li>
                    </ul>
                  </div>

                  <div>
                    <h2 className="text-lg font-semibold text-white mb-2">Architecture</h2>
                    <p>
                      The model combines advanced neural paradigms for comprehensive supply chain analysis with temporal and heterogeneous relationship modeling.
                    </p>
                  </div>

                  <div>
                    <h2 className="text-lg font-semibold text-white mb-2">Use Cases</h2>
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
      </div>

      {/* Footer */}
      <footer className="relative z-10 bg-black/80 backdrop-blur-sm border-t border-gray-700 text-gray-400 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          <p>HT-HGNN Supply Chain Risk Analysis Platform © 2026</p>
          <p className="mt-1">Powered by PyTorch and React</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
