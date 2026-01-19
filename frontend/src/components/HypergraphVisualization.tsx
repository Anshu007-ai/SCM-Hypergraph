import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { Zap, AlertTriangle, Maximize2, Play, Pause } from 'lucide-react';

type HypergraphNode = {
  id: string;
  label?: string;
  reliability: number;
  tier: number;
  affects: string[];
};

type Props = {
  nodes: HypergraphNode[];
  onNodeSelect?: (node: HypergraphNode) => void;
};

export const HypergraphVisualization: React.FC<Props> = ({ nodes, onNodeSelect }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const pulseStateRef = useRef<Map<string, { time: number; maxTime: number }>>(new Map());
  
  const [disrupted, setDisrupted] = useState<string | null>(null);
  const [impacted, setImpacted] = useState<Set<string>>(new Set());
  const [physicsEnabled, setPhysicsEnabled] = useState(true);

  // Build graph data from hypergraph nodes
  const { nodesData, edgesData } = useMemo(() => {
    const nodesData = nodes.map((n) => {
      const baseColor = n.reliability > 0.9 ? '#10b981' : n.reliability > 0.8 ? '#3b82f6' : '#f59e0b';
      return {
        id: n.id,
        label: n.label || n.id.substring(0, 8),
        title: `${n.id}\nReliability: ${(n.reliability * 100).toFixed(1)}%\nTier: ${n.tier}`,
        color: {
          background: baseColor,
          border: baseColor,
          highlight: { background: baseColor, border: '#1f2937' },
        },
        borderWidth: 2,
        borderWidthSelected: 4,
        font: { 
          size: 14, 
          color: '#ffffff',
          face: 'Inter, system-ui, sans-serif',
          bold: { color: '#ffffff' } 
        },
        size: 25,
        shape: 'dot',
        physics: true,
        shadow: {
          enabled: true,
          color: 'rgba(0,0,0,0.2)',
          size: 10,
          x: 2,
          y: 2
        }
      };
    });

    const edgesData = nodes
      .flatMap((n) => n.affects.map((t) => ({ from: n.id, to: t })))
      .map((e, i) => ({
        id: `edge-${i}`,
        from: e.from,
        to: e.to,
        arrows: {
          to: { enabled: true, scaleFactor: 0.8 }
        },
        color: { 
          color: 'rgba(156, 163, 175, 0.4)', 
          highlight: '#ef4444',
          hover: 'rgba(156, 163, 175, 0.7)'
        },
        width: 2,
        smooth: {
          enabled: true,
          type: 'cubicBezier',
          roundness: 0.5
        },
        physics: true,
      }));

    return { nodesData, edgesData };
  }, [nodes]);

  // Compute impacted nodes (BFS)
  const computeImpacted = (startId: string) => {
    const q = [startId];
    const vis = new Set<string>();
    while (q.length) {
      const cur = q.shift()!;
      if (vis.has(cur)) continue;
      vis.add(cur);
      const n = nodes.find((x) => x.id === cur);
      if (!n) continue;
      n.affects.forEach((a) => {
        if (!vis.has(a)) q.push(a);
      });
    }
    return vis;
  };

  // Animate pulse using requestAnimationFrame for smoother performance
  const animatePulse = () => {
    if (!networkRef.current || pulseStateRef.current.size === 0) return;

    const network = networkRef.current;
    const now = Date.now();

    pulseStateRef.current.forEach((state, nodeId) => {
      const elapsed = now - state.time;
      const progress = Math.min(elapsed / state.maxTime, 1);

      // Easing function for smooth pulse
      const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
      const easedProgress = easeOutCubic(progress);

      // Calculate scale (1 -> 1.5 -> 1)
      const scale = progress < 0.5 
        ? 1 + easedProgress * 1.0
        : 2 - easedProgress * 1.0;

      try {
        const node = (network as any).body.nodes[nodeId];
        if (node) {
          node.setOptions({
            size: 25 * scale,
            borderWidth: 2 + scale * 2,
          });
        }
      } catch (e) {
        // Ignore errors during animation
      }

      if (progress >= 1) {
        pulseStateRef.current.delete(nodeId);
      }
    });

    if (pulseStateRef.current.size > 0) {
      animationFrameRef.current = requestAnimationFrame(animatePulse);
    }
  };

  const animateImpact = (impactedSet: Set<string>) => {
    if (!networkRef.current) return;

    const network = networkRef.current;
    const now = Date.now();

    // Initialize pulse state for all impacted nodes
    impactedSet.forEach((id) => {
      pulseStateRef.current.set(id, { time: now, maxTime: 1200 });
    });

    // Start animation loop
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    animationFrameRef.current = requestAnimationFrame(animatePulse);

    // Highlight affected edges
    const edges = (network as any).body.data.edges;
    edges.forEach((edge: any) => {
      const edgeData = edges.get(edge.id);
      if (impactedSet.has(edgeData.from) && impactedSet.has(edgeData.to)) {
        edges.update({
          id: edge.id,
          color: { color: '#ef4444' },
          width: 4
        });
      }
    });
  };

  // Initialize vis-network
  useEffect(() => {
    if (!containerRef.current) return;

    const options = {
      physics: {
        enabled: physicsEnabled,
        barnesHut: {
          gravitationalConstant: -30000,
          centralGravity: 0.3,
          springLength: 200,
          springConstant: 0.05,
          damping: 0.09,
          avoidOverlap: 0.2
        },
        maxVelocity: 50,
        minVelocity: 0.75,
        stabilization: { 
          iterations: 300,
          updateInterval: 25
        },
      },
      interaction: {
        navigationButtons: true,
        keyboard: true,
        zoomView: true,
        dragView: true,
        hover: true,
        tooltipDelay: 200,
        hideEdgesOnDrag: true,
        hideEdgesOnZoom: true
      },
      layout: {
        randomSeed: 42,
        improvedLayout: true,
      },
      edges: {
        smooth: {
          enabled: true,
          type: 'cubicBezier',
          forceDirection: 'horizontal',
          roundness: 0.5
        }
      }
    };

    const data = {
      nodes: nodesData,
      edges: edgesData,
    };

    networkRef.current = new Network(containerRef.current, data, options);

    networkRef.current.on('click', (params: any) => {
      const nodeId = params.nodes?.[0];
      if (!nodeId) {
        // Reset on background click
        handleReset();
        return;
      }

      const impactedSet = computeImpacted(nodeId);
      setDisrupted(nodeId);
      setImpacted(impactedSet);
      animateImpact(impactedSet);

      const selectedNode = nodes.find((n) => n.id === nodeId);
      if (selectedNode) onNodeSelect?.(selectedNode);

      // Focus on node with smooth animation
      networkRef.current?.focus(nodeId, { 
        scale: 1.2, 
        animation: { 
          duration: 600, 
          easingFunction: 'easeInOutCubic' 
        } 
      });
    });

    // Hover effects
    networkRef.current.on('hoverNode', () => {
      containerRef.current!.style.cursor = 'pointer';
    });

    networkRef.current.on('blurNode', () => {
      containerRef.current!.style.cursor = 'default';
    });

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      networkRef.current?.destroy();
    };
  }, [nodesData, edgesData, physicsEnabled, nodes]);

  // Update physics when toggled
  useEffect(() => {
    if (networkRef.current) {
      networkRef.current.setOptions({ physics: { enabled: physicsEnabled } });
    }
  }, [physicsEnabled]);

  const handleReset = () => {
    setDisrupted(null);
    setImpacted(new Set());
    pulseStateRef.current.clear();
    
    if (networkRef.current) {
      // Reset all edges to default color
      const edges = (networkRef.current as any).body.data.edges;
      edges.forEach((edge: any) => {
        edges.update({
          id: edge.id,
          color: { color: 'rgba(156, 163, 175, 0.4)' },
          width: 2
        });
      });
      
      // Reset all nodes to default size
      const nodesCollection = (networkRef.current as any).body.data.nodes;
      nodesCollection.forEach((node: any) => {
        nodesCollection.update({
          id: node.id,
          size: 25,
          borderWidth: 2
        });
      });
      
      networkRef.current.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Zap className="w-5 h-5 text-blue-600" />
          Hypergraph Network Visualization
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => networkRef.current?.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } })}
            className="px-3 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-50 transition-colors flex items-center gap-1.5 text-sm font-medium shadow-sm"
            title="Fit to view"
          >
            <Maximize2 className="w-4 h-4" />
            Fit View
          </button>
          <button
            onClick={() => setPhysicsEnabled((s) => !s)}
            className="px-3 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-50 transition-colors flex items-center gap-1.5 text-sm font-medium shadow-sm"
            title="Toggle physics simulation"
          >
            {physicsEnabled ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {physicsEnabled ? 'Pause' : 'Resume'}
          </button>
          <button
            onClick={handleReset}
            disabled={!disrupted}
            className="px-3 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium shadow-sm"
            title="Reset visualization"
          >
            Reset
          </button>
        </div>
      </div>

      <div 
        ref={containerRef} 
        className="w-full h-[700px] border-2 border-gray-200 rounded-xl bg-gradient-to-br from-gray-50 to-white shadow-lg overflow-hidden" 
      />

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg border border-green-200">
          <div className="w-4 h-4 rounded-full bg-green-500 shadow-sm"></div>
          <span className="text-gray-700 font-medium">Reliable (&gt;90%)</span>
        </div>
        <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="w-4 h-4 rounded-full bg-blue-500 shadow-sm"></div>
          <span className="text-gray-700 font-medium">Good (80-90%)</span>
        </div>
        <div className="flex items-center gap-2 p-3 bg-amber-50 rounded-lg border border-amber-200">
          <div className="w-4 h-4 rounded-full bg-amber-500 shadow-sm"></div>
          <span className="text-gray-700 font-medium">At Risk (&lt;80%)</span>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-900 mb-2">
          💡 <strong>Click on a node</strong> to simulate disruption and see impact cascades through the network
        </p>
        <p className="text-sm text-blue-800">
          🖱️ <strong>Drag nodes</strong> to reorganize • <strong>Scroll</strong> to zoom • <strong>Click background</strong> to reset
        </p>
      </div>

      {disrupted && (
        <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4 shadow-md animate-in fade-in slide-in-from-top-2">
          <p className="text-base font-semibold text-red-900 flex items-center gap-2 mb-3">
            <AlertTriangle className="w-5 h-5" />
            Disruption Impact Analysis
          </p>
          <p className="text-sm text-red-800 mb-3">
            Disruption at <span className="font-bold bg-red-100 px-2 py-0.5 rounded">{disrupted}</span> affects{' '}
            <span className="font-bold">{impacted.size}</span> node(s) in the network.
          </p>
          <div className="bg-white rounded p-3 border border-red-200">
            <p className="text-xs font-semibold text-red-900 mb-2">Affected nodes:</p>
            <p className="text-xs text-red-700 max-h-24 overflow-y-auto font-mono">
              {Array.from(impacted).join(', ')}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};