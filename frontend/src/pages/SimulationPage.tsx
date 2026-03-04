import React, { useMemo } from 'react';
import { AlertTriangle } from 'lucide-react';
import { CascadeSimulator } from '../components/CascadeSimulator';
import { HypergraphCanvas } from '../components/HypergraphCanvas';
import { useCascade } from '../hooks/useCascade';
import type { HypergraphData, CascadeStep } from '../types';

interface SimulationPageProps {
  graphData: HypergraphData | null;
}

export const SimulationPage: React.FC<SimulationPageProps> = ({ graphData }) => {
  const {
    cascadeResult,
    isSimulating,
    currentStep,
    currentStepData,
    error,
    selectedShockNodes,
    shockMagnitude,
    startSimulation,
    resetSimulation,
    addShockNode,
    removeShockNode,
    setShockMagnitude,
    stepForward,
    stepBackward,
    playAnimation,
    pauseAnimation,
  } = useCascade();

  // Build the cascade overlay from the current step so the canvas can highlight
  // disrupted / at-risk nodes.
  const cascadeOverlay = useMemo(() => {
    if (!currentStepData) return undefined;

    // Collect all disrupted nodes up to (and including) the current step
    const disrupted = new Set<string>();
    if (cascadeResult?.timeline) {
      for (let i = 0; i <= currentStep; i++) {
        const step: CascadeStep = cascadeResult.timeline[i];
        step.newlyDisrupted.forEach((id) => disrupted.add(id));
      }
    }
    const atRisk = new Set<string>(currentStepData.atRisk);

    return { disruptedNodes: disrupted, atRiskNodes: atRisk };
  }, [cascadeResult, currentStep, currentStepData]);

  const handleStartSimulation = () => {
    if (selectedShockNodes.length === 0) return;
    startSimulation({
      shockNodes: selectedShockNodes,
      shockMagnitude,
      propagationSteps: 10,
      cascadeThreshold: 0.3,
    });
  };

  if (!graphData) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-12 text-center">
          <AlertTriangle className="w-10 h-10 text-gray-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">No Graph Loaded</h2>
          <p className="text-gray-400 text-sm">
            Go to the Dashboard and load a dataset before running cascade simulations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Cascade Simulation</h1>
        <p className="text-gray-400 text-sm">
          Select shock nodes and run cascade propagation to see how disruptions spread
          through the supply chain hypergraph.
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
        {/* Left Panel: Cascade Controls */}
        <div className="lg:col-span-1">
          <CascadeSimulator
            graphData={graphData}
            cascadeResult={cascadeResult}
            currentStep={currentStep}
            isSimulating={isSimulating}
            shockNodes={selectedShockNodes}
            shockMagnitude={shockMagnitude}
            onAddShockNode={addShockNode}
            onRemoveShockNode={removeShockNode}
            onSetShockMagnitude={setShockMagnitude}
            onStartSimulation={handleStartSimulation}
            onResetSimulation={resetSimulation}
            onStepForward={stepForward}
            onStepBackward={stepBackward}
            onPlayAnimation={playAnimation}
            onPauseAnimation={pauseAnimation}
          />
        </div>

        {/* Right Panel: Hypergraph Canvas */}
        <div className="lg:col-span-2">
          <HypergraphCanvas
            graphData={graphData}
            selectedNodeId={null}
            selectedHyperedgeId={null}
            onSelectNode={(nodeId) => {
              // When a node is clicked on the canvas, add it as a shock source
              if (!selectedShockNodes.includes(nodeId)) {
                addShockNode(nodeId);
              }
            }}
            onSelectHyperedge={() => {}}
            cascadeState={cascadeOverlay}
          />
        </div>
      </div>
    </div>
  );
};
