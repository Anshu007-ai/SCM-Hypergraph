import React, { useState, useMemo } from 'react';
import { Search, AlertTriangle, Brain } from 'lucide-react';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';
import { useExplain } from '../hooks/useExplain';
import type { HypergraphData, ExplainRequest } from '../types';

interface ExplainPageProps {
  graphData: HypergraphData | null;
}

export const ExplainPage: React.FC<ExplainPageProps> = ({ graphData }) => {
  const {
    explanations,
    isExplaining,
    error,
    selectedNodeId,
    selectedExplanation,
    explainNode,
    selectNode,
    clearExplanations,
  } = useExplain();

  const [searchQuery, setSearchQuery] = useState('');
  const [predictionType, setPredictionType] =
    useState<ExplainRequest['predictionType']>('criticality');

  // Filter nodes for the left-panel list
  const filteredNodes = useMemo(() => {
    if (!graphData) return [];
    const query = searchQuery.toLowerCase();
    return graphData.nodes.filter(
      (n) =>
        n.id.toLowerCase().includes(query) ||
        n.label.toLowerCase().includes(query),
    );
  }, [graphData, searchQuery]);

  const handleSelectNode = (nodeId: string) => {
    selectNode(nodeId);
    // Auto-request explanation when a node is selected
    explainNode(nodeId, predictionType);
  };

  const handleRequestExplanation = (nodeId: string) => {
    explainNode(nodeId, predictionType);
  };

  if (!graphData) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-12 text-center">
          <AlertTriangle className="w-10 h-10 text-gray-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">No Graph Loaded</h2>
          <p className="text-gray-400 text-sm">
            Go to the Dashboard and load a dataset before exploring explanations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Explainability</h1>
        <p className="text-gray-400 text-sm">
          Select a node to generate HyperSHAP explanations showing which
          hyperedges and features contribute most to the predicted risk.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Two-Column Layout */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left Panel: Node Selector */}
        <div className="lg:col-span-1 space-y-4">
          {/* Search & Prediction Type */}
          <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-semibold text-white">Node Selector</h3>
            </div>

            {/* Prediction Type Toggle */}
            <div className="mb-4">
              <label className="block text-xs text-gray-400 mb-2 font-medium">
                Prediction Type
              </label>
              <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
                {(['criticality', 'price', 'change'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setPredictionType(type)}
                    className={`flex-1 px-3 py-1.5 rounded text-xs font-medium transition-all duration-200 capitalize ${
                      predictionType === type
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'text-gray-400 hover:text-gray-300'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Search Input */}
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search nodes..."
                className="w-full pl-9 pr-3 py-2 rounded-lg text-sm bg-gray-900 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 transition-colors"
              />
            </div>

            {/* Node count */}
            <p className="text-[10px] text-gray-500 mb-2">
              {filteredNodes.length} of {graphData.nodes.length} nodes
            </p>
          </div>

          {/* Scrollable Node List */}
          <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm overflow-hidden">
            <div className="max-h-[500px] overflow-y-auto divide-y divide-gray-700/50">
              {filteredNodes.length === 0 && (
                <div className="p-6 text-center">
                  <p className="text-gray-500 text-xs">No nodes match your search.</p>
                </div>
              )}
              {filteredNodes.map((node) => {
                const isSelected = selectedNodeId === node.id;
                const hasExplanation = explanations.has(node.id);
                return (
                  <button
                    key={node.id}
                    onClick={() => handleSelectNode(node.id)}
                    className={`w-full text-left px-4 py-3 transition-colors text-sm ${
                      isSelected
                        ? 'bg-purple-500/10 border-l-2 border-purple-500'
                        : 'hover:bg-gray-800/60 border-l-2 border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`font-medium truncate ${
                          isSelected ? 'text-purple-300' : 'text-gray-200'
                        }`}
                      >
                        {node.label}
                      </span>
                      <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                        {hasExplanation && (
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500" title="Explanation loaded" />
                        )}
                        {node.riskLevel && (
                          <span
                            className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                            style={{
                              color:
                                node.riskLevel === 'CRITICAL'
                                  ? '#fca5a5'
                                  : node.riskLevel === 'HIGH'
                                  ? '#fdba74'
                                  : node.riskLevel === 'MEDIUM'
                                  ? '#fde047'
                                  : '#86efac',
                              backgroundColor:
                                node.riskLevel === 'CRITICAL'
                                  ? 'rgba(239,68,68,0.15)'
                                  : node.riskLevel === 'HIGH'
                                  ? 'rgba(249,115,22,0.15)'
                                  : node.riskLevel === 'MEDIUM'
                                  ? 'rgba(234,179,8,0.15)'
                                  : 'rgba(34,197,94,0.15)',
                            }}
                          >
                            {node.riskLevel}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-[10px] text-gray-500 mt-0.5 font-mono truncate">
                      {node.id}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Clear Button */}
          {explanations.size > 0 && (
            <button
              onClick={clearExplanations}
              className="w-full px-4 py-2.5 rounded-xl text-sm font-medium border border-gray-700 hover:bg-gray-800 text-gray-300 transition-colors"
            >
              Clear All Explanations
            </button>
          )}
        </div>

        {/* Right Panel: ExplainabilityPanel */}
        <div className="lg:col-span-2">
          <ExplainabilityPanel
            explanation={selectedExplanation}
            isLoading={isExplaining}
            onRequestExplanation={handleRequestExplanation}
          />
        </div>
      </div>
    </div>
  );
};
