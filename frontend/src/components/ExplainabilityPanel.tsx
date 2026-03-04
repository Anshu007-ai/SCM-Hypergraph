import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import {
  Brain,
  Loader2,
  Search,
  Info,
  TrendingUp,
  Layers,
} from 'lucide-react';
import type { NodeExplanation } from '../types';

interface ExplainabilityPanelProps {
  explanation: NodeExplanation | null;
  isLoading: boolean;
  onRequestExplanation: (nodeId: string) => void;
}

const RISK_BADGE_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  LOW: { bg: 'bg-green-500/15', text: 'text-green-400', border: 'border-green-500/30' },
  MEDIUM: { bg: 'bg-yellow-500/15', text: 'text-yellow-400', border: 'border-yellow-500/30' },
  HIGH: { bg: 'bg-orange-500/15', text: 'text-orange-400', border: 'border-orange-500/30' },
  CRITICAL: { bg: 'bg-red-500/15', text: 'text-red-400', border: 'border-red-500/30' },
};

function getRiskBadge(risk: string) {
  const style = RISK_BADGE_STYLES[risk] ?? {
    bg: 'bg-gray-500/15',
    text: 'text-gray-400',
    border: 'border-gray-500/30',
  };
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${style.bg} ${style.text} ${style.border}`}
    >
      {risk}
    </span>
  );
}

/** Custom tooltip for SHAP charts */
const ShapTooltip: React.FC<{
  active?: boolean;
  payload?: Array<{ value: number; payload: { name: string; shapValue: number } }>;
  label?: string;
}> = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-900 p-2 shadow-xl text-xs">
      <p className="text-gray-300 font-medium mb-1">{data.name}</p>
      <p className={data.shapValue >= 0 ? 'text-blue-400' : 'text-red-400'}>
        SHAP: {data.shapValue >= 0 ? '+' : ''}
        {data.shapValue.toFixed(4)}
      </p>
    </div>
  );
};

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  explanation,
  isLoading,
  onRequestExplanation,
}) => {
  const [nodeIdInput, setNodeIdInput] = useState('');

  // Prepare hyperedge attribution data sorted by absolute SHAP value
  const hyperedgeData = useMemo(() => {
    if (!explanation) return [];
    return [...explanation.hyperedgeAttributions]
      .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
      .map((attr) => ({
        name: attr.edgeId.length > 14 ? attr.edgeId.slice(0, 14) + '...' : attr.edgeId,
        fullName: attr.edgeId,
        shapValue: attr.shapValue,
        type: attr.type,
      }));
  }, [explanation]);

  // Prepare feature attribution data sorted by absolute SHAP value
  const featureData = useMemo(() => {
    if (!explanation) return [];
    return [...explanation.featureAttributions]
      .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
      .map((attr) => ({
        name: attr.feature.length > 16 ? attr.feature.slice(0, 16) + '...' : attr.feature,
        fullName: attr.feature,
        shapValue: attr.shapValue,
      }));
  }, [explanation]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = nodeIdInput.trim();
    if (trimmed) {
      onRequestExplanation(trimmed);
    }
  };

  return (
    <div className="space-y-4">
      {/* Explain Node Input */}
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-semibold text-white">HyperSHAP Explainability</h3>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
            <input
              type="text"
              value={nodeIdInput}
              onChange={(e) => setNodeIdInput(e.target.value)}
              placeholder="Enter node ID to explain..."
              className="w-full pl-9 pr-3 py-2.5 rounded-lg text-sm bg-gray-900 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 transition-colors"
            />
          </div>
          <button
            type="submit"
            disabled={!nodeIdInput.trim() || isLoading}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/20"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Brain className="w-4 h-4" />
            )}
            Explain
          </button>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-8 text-center">
          <Loader2 className="w-8 h-8 text-purple-400 mx-auto mb-3 animate-spin" />
          <p className="text-gray-400 text-sm">Computing SHAP explanations...</p>
          <p className="text-gray-500 text-xs mt-1">This may take a moment for large graphs.</p>
        </div>
      )}

      {/* No Explanation State */}
      {!isLoading && !explanation && (
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-8 text-center">
          <Brain className="w-8 h-8 text-gray-500 mx-auto mb-3" />
          <p className="text-gray-400 text-sm">No explanation loaded.</p>
          <p className="text-gray-500 text-xs mt-1">
            Enter a node ID above and click "Explain" to generate SHAP attributions.
          </p>
        </div>
      )}

      {/* Explanation Content */}
      {!isLoading && explanation && (
        <>
          {/* Node Info Header */}
          <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-1">
                  Node
                </p>
                <p className="text-lg font-bold text-white font-mono">
                  {explanation.nodeId}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-1">
                  Predicted Risk
                </p>
                {getRiskBadge(explanation.predictedRisk)}
              </div>
            </div>

            {/* Confidence Bar */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-gray-400 font-medium">Confidence</span>
                <span className="text-xs font-mono text-white">
                  {(explanation.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${explanation.confidence * 100}%`,
                    background:
                      explanation.confidence >= 0.8
                        ? 'linear-gradient(90deg, #22c55e, #16a34a)'
                        : explanation.confidence >= 0.5
                        ? 'linear-gradient(90deg, #eab308, #ca8a04)'
                        : 'linear-gradient(90deg, #ef4444, #dc2626)',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Hyperedge Attributions */}
          {hyperedgeData.length > 0 && (
            <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-semibold text-white">
                  Hyperedge Attributions
                </h3>
                <span className="text-[10px] text-gray-500 ml-auto">
                  Sorted by |SHAP|
                </span>
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={hyperedgeData}
                    layout="vertical"
                    margin={{ top: 5, right: 20, bottom: 5, left: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                      axisLine={{ stroke: '#4b5563' }}
                      tickLine={{ stroke: '#4b5563' }}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                      axisLine={{ stroke: '#4b5563' }}
                      tickLine={false}
                      width={110}
                    />
                    <Tooltip content={<ShapTooltip />} />
                    <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1} />
                    <Bar dataKey="shapValue" radius={[0, 4, 4, 0]} maxBarSize={18}>
                      {hyperedgeData.map((entry, index) => (
                        <Cell
                          key={`cell-he-${index}`}
                          fill={entry.shapValue >= 0 ? '#3b82f6' : '#ef4444'}
                          fillOpacity={0.8}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Feature Attributions */}
          {featureData.length > 0 && (
            <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-white">
                  Feature Attributions
                </h3>
                <span className="text-[10px] text-gray-500 ml-auto">
                  Sorted by |SHAP|
                </span>
              </div>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={featureData}
                    layout="vertical"
                    margin={{ top: 5, right: 20, bottom: 5, left: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                      axisLine={{ stroke: '#4b5563' }}
                      tickLine={{ stroke: '#4b5563' }}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: '#9ca3af', fontSize: 10 }}
                      axisLine={{ stroke: '#4b5563' }}
                      tickLine={false}
                      width={120}
                    />
                    <Tooltip content={<ShapTooltip />} />
                    <ReferenceLine x={0} stroke="#6b7280" strokeWidth={1} />
                    <Bar dataKey="shapValue" radius={[0, 4, 4, 0]} maxBarSize={18}>
                      {featureData.map((entry, index) => (
                        <Cell
                          key={`cell-feat-${index}`}
                          fill={entry.shapValue >= 0 ? '#3b82f6' : '#ef4444'}
                          fillOpacity={0.8}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Color Legend */}
              <div className="flex items-center justify-center gap-6 mt-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm bg-blue-500" />
                  <span className="text-gray-400">Positive SHAP (increases risk)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm bg-red-500" />
                  <span className="text-gray-400">Negative SHAP (decreases risk)</span>
                </div>
              </div>
            </div>
          )}

          {/* Recommendation */}
          {explanation.recommendation && (
            <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
              <div className="flex items-center gap-2 mb-3">
                <Info className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Recommendation</h3>
              </div>
              <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-4">
                <p className="text-sm text-gray-300 leading-relaxed">
                  {explanation.recommendation}
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
