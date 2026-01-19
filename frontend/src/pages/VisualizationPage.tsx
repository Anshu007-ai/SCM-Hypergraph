import React, { useState } from 'react';
import { HypergraphVisualization } from '../components/HypergraphVisualization';
import { RealTimeStream } from '../components/RealTimeStream';
import type { RiskAssessment } from '../types';

interface VisualizationPageProps {
  riskAssessments: RiskAssessment[];
}

interface DataStreamEvent {
  nodeId: string;
  timestamp: string;
  priceChange: number;
  percentChange: number;
  reason: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export const VisualizationPage: React.FC<VisualizationPageProps> = ({ riskAssessments }) => {
  const [selectedNode, setSelectedNode] = useState<RiskAssessment | null>(null);
  const [impactLog, setImpactLog] = useState<DataStreamEvent[]>([]);

  // Convert risk assessments to hypergraph nodes
  const hypergraphNodes = riskAssessments.map((ra, idx) => {
    const nextNodes = riskAssessments
      .slice(idx + 1)
      .filter(() => Math.random() > 0.5)
      .map((n) => n.nodeId);

    return {
      id: ra.nodeId,
      label: ra.nodeId,
      cost: Math.random() * 100,
      reliability: ra.criticalityScore,
      leadTime: Math.random() * 30,
      tier: Math.ceil(Math.random() * 4),
      affectedBy: [],
      affects: nextNodes,
    };
  });

  const handlePriceDisruption = (event: DataStreamEvent) => {
    setImpactLog((prev) => [event, ...prev.slice(0, 9)]);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg p-8">
        <h1 className="text-3xl font-bold mb-2">Hypergraph Network Visualization</h1>
        <p className="text-purple-100">
          Real-time supply chain disruption analysis with cascading impact propagation
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Hypergraph Visualization */}
        <div className="lg:col-span-2">
          <HypergraphVisualization
            nodes={hypergraphNodes}
            onNodeSelect={(node) => {
              const matched = riskAssessments.find((ra) => ra.nodeId === node.id);
              setSelectedNode(matched || null);
            }}
          />
        </div>

        {/* Right Sidebar */}
        <div className="space-y-4">
          {/* Selected Node Details */}
          {selectedNode && (
            <div className="bg-white border border-gray-300 rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-3">Node Details</h3>
              <div className="space-y-2 text-sm">
                <div>
                  <p className="text-gray-600">Node ID</p>
                  <p className="font-mono font-semibold text-gray-900">{selectedNode.nodeId}</p>
                </div>
                <div>
                  <p className="text-gray-600">Risk Level</p>
                  <div
                    className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                      selectedNode.riskLevel === 'CRITICAL'
                        ? 'bg-red-100 text-red-900'
                        : selectedNode.riskLevel === 'HIGH'
                          ? 'bg-orange-100 text-orange-900'
                          : selectedNode.riskLevel === 'MEDIUM'
                            ? 'bg-yellow-100 text-yellow-900'
                            : 'bg-green-100 text-green-900'
                    }`}
                  >
                    {selectedNode.riskLevel}
                  </div>
                </div>
                <div>
                  <p className="text-gray-600">Criticality Score</p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-red-500 h-2 rounded-full transition-all"
                      style={{ width: `${selectedNode.criticalityScore * 100}%` }}
                    ></div>
                  </div>
                  <p className="font-semibold text-gray-900 mt-1">
                    {(selectedNode.criticalityScore * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-600">Price Prediction</p>
                  <p className="font-semibold text-gray-900">${selectedNode.pricePrediction.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Price Change</p>
                  <p
                    className={`font-semibold ${
                      selectedNode.changeForecasted > 0 ? 'text-red-600' : 'text-green-600'
                    }`}
                  >
                    {selectedNode.changeForecasted > 0 ? '+' : ''}
                    {(selectedNode.changeForecasted * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {/* Recommendations */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="font-semibold text-gray-900 mb-2 text-sm">Recommendations</p>
                <ul className="space-y-1 text-xs text-gray-700">
                  {selectedNode.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="text-blue-600 font-bold">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {!selectedNode && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-900">
                Select a node in the hypergraph visualization to view its details and risk assessment.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Real-Time Stream */}
      <div className="bg-white border border-gray-300 rounded-lg p-6 shadow-sm">
        <RealTimeStream onPriceDisruption={handlePriceDisruption} />
      </div>

      {/* Impact Log */}
      {impactLog.length > 0 && (
        <div className="bg-white border border-gray-300 rounded-lg p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Impact Log</h3>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {impactLog.map((event, idx) => (
              <div key={idx} className="flex items-start justify-between p-3 bg-gray-50 rounded-lg text-sm">
                <div>
                  <p className="font-semibold text-gray-900">{event.nodeId}</p>
                  <p className="text-xs text-gray-600 mt-1">{event.reason}</p>
                </div>
                <div className="text-right">
                  <p className={`font-bold ${event.priceChange > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {event.priceChange > 0 ? '+' : ''}${event.priceChange.toFixed(0)}
                  </p>
                  <p className="text-xs text-gray-600">{event.timestamp}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="grid md:grid-cols-2 gap-4 text-sm bg-gray-50 rounded-lg p-6">
        <div>
          <p className="font-semibold text-gray-900 mb-3">Node Colors (by Reliability)</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-green-500"></div>
              <span className="text-gray-600">Highly Reliable (&gt;90%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-blue-500"></div>
              <span className="text-gray-600">Good Reliability (80-90%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-amber-500"></div>
              <span className="text-gray-600">At Risk (&lt;80%)</span>
            </div>
          </div>
        </div>
        <div>
          <p className="font-semibold text-gray-900 mb-3">How to Use</p>
          <ul className="space-y-1 text-gray-600">
            <li>• <strong>Click nodes</strong> to simulate disruptions and see cascading effects</li>
            <li>• <strong>Use Stream controls</strong> to replay real-time supply chain events</li>
            <li>• <strong>View Impact Log</strong> to track all disruptions over time</li>
            <li>• <strong>Check recommendations</strong> for mitigation strategies</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
