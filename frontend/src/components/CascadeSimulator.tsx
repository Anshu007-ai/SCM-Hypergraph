import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  RotateCcw,
  Zap,
  X,
  Search,
  AlertTriangle,
  Activity,
  Loader2,
} from 'lucide-react';
import type { HypergraphData, CascadeResult } from '../types';

interface CascadeSimulatorProps {
  graphData: HypergraphData | null;
  cascadeResult: CascadeResult | null;
  currentStep: number;
  isSimulating: boolean;
  shockNodes: string[];
  shockMagnitude: number;
  onAddShockNode: (nodeId: string) => void;
  onRemoveShockNode: (nodeId: string) => void;
  onSetShockMagnitude: (mag: number) => void;
  onStartSimulation: () => void;
  onResetSimulation: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onPlayAnimation: () => void;
  onPauseAnimation: () => void;
}

export const CascadeSimulator: React.FC<CascadeSimulatorProps> = ({
  graphData,
  cascadeResult,
  currentStep,
  isSimulating,
  shockNodes,
  shockMagnitude,
  onAddShockNode,
  onRemoveShockNode,
  onSetShockMagnitude,
  onStartSimulation,
  onResetSimulation,
  onStepForward,
  onStepBackward,
  onPlayAnimation,
  onPauseAnimation,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // Filter available nodes for the dropdown
  const filteredNodes = useMemo(() => {
    if (!graphData) return [];
    const query = searchQuery.toLowerCase();
    return graphData.nodes
      .filter(
        (n) =>
          !shockNodes.includes(n.id) &&
          (n.id.toLowerCase().includes(query) || n.label.toLowerCase().includes(query))
      )
      .slice(0, 20);
  }, [graphData, searchQuery, shockNodes]);

  // Build timeline chart data from cascade result
  const timelineData = useMemo(() => {
    if (!cascadeResult?.timeline) return [];
    return cascadeResult.timeline.map((step) => ({
      step: step.step,
      disrupted: step.disruptedCount,
      newlyDisrupted: step.newlyDisrupted.length,
      atRisk: step.atRisk.length,
    }));
  }, [cascadeResult]);

  // Compute stats
  const stats = useMemo(() => {
    if (!cascadeResult) {
      return { totalDisrupted: 0, cascadeSize: 0, criticalPaths: 0 };
    }
    return {
      totalDisrupted: cascadeResult.totalDisrupted,
      cascadeSize: cascadeResult.timeline.length,
      criticalPaths: cascadeResult.criticalPaths.length,
    };
  }, [cascadeResult]);

  const handleTogglePlay = () => {
    if (isPlaying) {
      onPauseAnimation();
    } else {
      onPlayAnimation();
    }
    setIsPlaying(!isPlaying);
  };

  if (!graphData) {
    return (
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-8 text-center">
        <AlertTriangle className="w-8 h-8 text-gray-500 mx-auto mb-3" />
        <p className="text-gray-400 text-sm">Load a dataset to run cascade simulations.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Shock Configuration */}
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-white">Shock Configuration</h3>
        </div>

        {/* Node Selector */}
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-2 font-medium">
            Shock Nodes
          </label>

          {/* Selected shock nodes */}
          {shockNodes.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {shockNodes.map((nodeId) => {
                const node = graphData.nodes.find((n) => n.id === nodeId);
                return (
                  <span
                    key={nodeId}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-red-500/20 text-red-300 text-xs border border-red-500/30"
                  >
                    {node?.label ?? nodeId}
                    <button
                      onClick={() => onRemoveShockNode(nodeId)}
                      className="hover:text-red-100 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                );
              })}
            </div>
          )}

          {/* Search dropdown */}
          <div className="relative">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setIsDropdownOpen(true);
                }}
                onFocus={() => setIsDropdownOpen(true)}
                onBlur={() => setTimeout(() => setIsDropdownOpen(false), 200)}
                placeholder="Search nodes to add..."
                className="w-full pl-9 pr-3 py-2 rounded-lg text-sm bg-gray-900 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition-colors"
              />
            </div>

            {isDropdownOpen && filteredNodes.length > 0 && (
              <div className="absolute z-20 mt-1 w-full max-h-48 overflow-y-auto rounded-lg border border-gray-700 bg-gray-900 shadow-xl">
                {filteredNodes.map((node) => (
                  <button
                    key={node.id}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      onAddShockNode(node.id);
                      setSearchQuery('');
                      setIsDropdownOpen(false);
                    }}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-gray-800 transition-colors flex items-center justify-between"
                  >
                    <span className="text-gray-200 truncate">{node.label}</span>
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
                      {node.riskLevel ?? 'N/A'}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {isDropdownOpen && filteredNodes.length === 0 && searchQuery && (
              <div className="absolute z-20 mt-1 w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-center">
                <p className="text-xs text-gray-500">No matching nodes found.</p>
              </div>
            )}
          </div>
        </div>

        {/* Magnitude Slider */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs text-gray-400 font-medium">Shock Magnitude</label>
            <span className="text-xs font-mono text-white bg-gray-800 px-2 py-0.5 rounded">
              {shockMagnitude.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={shockMagnitude}
            onChange={(e) => onSetShockMagnitude(parseFloat(e.target.value))}
            className="w-full h-1.5 rounded-full appearance-none bg-gray-700 accent-blue-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-gray-500 mt-1">
            <span>0.00 (minimal)</span>
            <span>1.00 (total)</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <button
            onClick={onStartSimulation}
            disabled={shockNodes.length === 0 || isSimulating}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-red-600 hover:bg-red-500 text-white"
          >
            {isSimulating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Simulating...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Start Simulation
              </>
            )}
          </button>
          <button
            onClick={() => {
              onResetSimulation();
              setIsPlaying(false);
            }}
            className="px-4 py-2.5 rounded-lg text-sm font-medium transition-colors border border-gray-700 hover:bg-gray-800 text-gray-300"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Playback Controls & Timeline */}
      {cascadeResult && (
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Cascade Timeline</h3>
          </div>

          {/* Playback Controls */}
          <div className="flex items-center justify-center gap-3 mb-4">
            <button
              onClick={onStepBackward}
              disabled={currentStep <= 0}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Previous step"
            >
              <SkipBack className="w-4 h-4" />
            </button>
            <button
              onClick={handleTogglePlay}
              disabled={!cascadeResult}
              className="p-3 rounded-full bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 transition-colors shadow-lg shadow-blue-600/20"
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button
              onClick={onStepForward}
              disabled={currentStep >= cascadeResult.timeline.length - 1}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="Next step"
            >
              <SkipForward className="w-4 h-4" />
            </button>
          </div>

          {/* Step Indicator */}
          <div className="flex items-center justify-between mb-4 text-xs">
            <span className="text-gray-400">
              Step{' '}
              <span className="text-white font-bold">
                {currentStep + 1}
              </span>{' '}
              of {cascadeResult.timeline.length}
            </span>
            <span className="text-gray-500">
              {cascadeResult.timeline[currentStep]?.disruptedCount ?? 0} disrupted
            </span>
          </div>

          {/* Progress Bar */}
          <div className="w-full h-1.5 bg-gray-800 rounded-full mb-5 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-red-500 rounded-full transition-all duration-300"
              style={{
                width: `${
                  cascadeResult.timeline.length > 1
                    ? (currentStep / (cascadeResult.timeline.length - 1)) * 100
                    : 100
                }%`,
              }}
            />
          </div>

          {/* Timeline Chart */}
          {timelineData.length > 0 && (
            <div className="h-48 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="step"
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickLine={{ stroke: '#4b5563' }}
                    label={{
                      value: 'Step',
                      position: 'insideBottom',
                      offset: -5,
                      fill: '#6b7280',
                      fontSize: 10,
                    }}
                  />
                  <YAxis
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    axisLine={{ stroke: '#4b5563' }}
                    tickLine={{ stroke: '#4b5563' }}
                    label={{
                      value: 'Count',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#6b7280',
                      fontSize: 10,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#111827',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#f3f4f6',
                    }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="disrupted"
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Total Disrupted"
                    dot={{ r: 3, fill: '#ef4444' }}
                    activeDot={{ r: 5 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="atRisk"
                    stroke="#f97316"
                    strokeWidth={1.5}
                    name="At Risk"
                    dot={{ r: 2, fill: '#f97316' }}
                    strokeDasharray="4 4"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Current Step Reference Line */}
          {cascadeResult.timeline[currentStep] && (
            <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3 text-xs">
              <p className="text-gray-400 mb-1">
                Newly disrupted at step {currentStep + 1}:
              </p>
              <div className="flex flex-wrap gap-1">
                {cascadeResult.timeline[currentStep].newlyDisrupted.length > 0 ? (
                  cascadeResult.timeline[currentStep].newlyDisrupted.map((nodeId) => (
                    <span
                      key={nodeId}
                      className="inline-block px-1.5 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/20 font-mono"
                    >
                      {nodeId}
                    </span>
                  ))
                ) : (
                  <span className="text-gray-500 italic">No new disruptions</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stats Panel */}
      {cascadeResult && (
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <h3 className="text-sm font-semibold text-white">Simulation Results</h3>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 font-medium mb-1">
                Total Disrupted
              </p>
              <p className="text-2xl font-bold text-red-400">{stats.totalDisrupted}</p>
            </div>
            <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 font-medium mb-1">
                Cascade Steps
              </p>
              <p className="text-2xl font-bold text-amber-400">{stats.cascadeSize}</p>
            </div>
            <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3 text-center">
              <p className="text-[10px] uppercase tracking-wider text-gray-500 font-medium mb-1">
                Critical Paths
              </p>
              <p className="text-2xl font-bold text-orange-400">{stats.criticalPaths}</p>
            </div>
          </div>

          {/* Status Badge */}
          <div className="mt-3 flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                cascadeResult.status === 'COMPLETED'
                  ? 'bg-green-500'
                  : cascadeResult.status === 'RUNNING'
                  ? 'bg-amber-500 animate-pulse'
                  : cascadeResult.status === 'FAILED'
                  ? 'bg-red-500'
                  : 'bg-gray-500'
              }`}
            />
            <span className="text-xs text-gray-400">
              Status: <span className="text-white font-medium">{cascadeResult.status}</span>
            </span>
            <span className="text-xs text-gray-500 ml-auto">
              Task: {cascadeResult.taskId}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
