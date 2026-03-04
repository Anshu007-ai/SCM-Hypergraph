import React, { useRef, useEffect, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize2, Loader2 } from 'lucide-react';
import type { HypergraphData } from '../types';

interface CascadeOverlay {
  disruptedNodes: Set<string>;
  atRiskNodes: Set<string>;
}

interface HypergraphCanvasProps {
  graphData: HypergraphData | null;
  selectedNodeId: string | null;
  selectedHyperedgeId: string | null;
  onSelectNode: (nodeId: string) => void;
  onSelectHyperedge: (edgeId: string) => void;
  cascadeState?: CascadeOverlay;
}

const RISK_COLORS: Record<string, string> = {
  LOW: '#22c55e',
  MEDIUM: '#eab308',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
};

const RISK_COLOR_DEFAULT = '#6b7280';

function getRiskColor(level?: string): string {
  return (level && RISK_COLORS[level]) || RISK_COLOR_DEFAULT;
}

/** Compute the convex hull for a set of 2D points using Andrew's monotone chain. */
function convexHull(points: [number, number][]): [number, number][] {
  if (points.length < 3) return points;

  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower: [number, number][] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }

  const upper: [number, number][] = [];
  for (const p of sorted.reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

const EDGE_PALETTE = [
  'rgba(59, 130, 246, 0.15)',
  'rgba(168, 85, 247, 0.15)',
  'rgba(236, 72, 153, 0.15)',
  'rgba(20, 184, 166, 0.15)',
  'rgba(245, 158, 11, 0.15)',
  'rgba(99, 102, 241, 0.15)',
  'rgba(234, 179, 8, 0.15)',
  'rgba(34, 197, 94, 0.15)',
];

const EDGE_STROKE_PALETTE = [
  'rgba(59, 130, 246, 0.5)',
  'rgba(168, 85, 247, 0.5)',
  'rgba(236, 72, 153, 0.5)',
  'rgba(20, 184, 166, 0.5)',
  'rgba(245, 158, 11, 0.5)',
  'rgba(99, 102, 241, 0.5)',
  'rgba(234, 179, 8, 0.5)',
  'rgba(34, 197, 94, 0.5)',
];

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
  riskLevel?: string;
  riskScore?: number;
}

export const HypergraphCanvas: React.FC<HypergraphCanvasProps> = ({
  graphData,
  selectedNodeId,
  selectedHyperedgeId,
  onSelectNode,
  onSelectHyperedge,
  cascadeState,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, undefined> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Observe container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Main D3 rendering
  useEffect(() => {
    if (!svgRef.current || !graphData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { width, height } = dimensions;
    const nodes: SimNode[] = graphData.nodes.map((n) => ({
      id: n.id,
      label: n.label,
      type: n.type,
      riskLevel: n.riskLevel,
      riskScore: n.riskScore,
    }));

    const hyperedges = graphData.hyperedges;

    // Build a node-id lookup
    const nodeMap = new Map<string, SimNode>();
    nodes.forEach((n) => nodeMap.set(n.id, n));

    // Create link pairs for force simulation (from hyperedges)
    interface SimLink {
      source: string;
      target: string;
      edgeId: string;
    }
    const links: SimLink[] = [];
    hyperedges.forEach((he) => {
      for (let i = 0; i < he.nodeIds.length; i++) {
        for (let j = i + 1; j < he.nodeIds.length; j++) {
          if (nodeMap.has(he.nodeIds[i]) && nodeMap.has(he.nodeIds[j])) {
            links.push({
              source: he.nodeIds[i],
              target: he.nodeIds[j],
              edgeId: he.id,
            });
          }
        }
      }
    });

    // Force simulation
    const simulation = d3
      .forceSimulation<SimNode>(nodes)
      .force(
        'link',
        d3
          .forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(100)
          .strength(0.3)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(25));

    simulationRef.current = simulation;

    // Zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        mainGroup.attr('transform', event.transform.toString());
      });

    zoomRef.current = zoom;
    svg.call(zoom);

    const mainGroup = svg.append('g');

    // Defs for filters and animations
    const defs = svg.append('defs');

    // Glow filter for selected node
    const glowFilter = defs.append('filter').attr('id', 'glow');
    glowFilter
      .append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');
    const feMerge = glowFilter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Pulse animation style
    const style = svg.append('style');
    style.text(`
      @keyframes pulse-risk {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
      }
      .at-risk-pulse {
        animation: pulse-risk 1.2s ease-in-out infinite;
      }
    `);

    // Draw hyperedge hulls
    const hullGroup = mainGroup.append('g').attr('class', 'hulls');

    function updateHulls() {
      hullGroup.selectAll('path').remove();

      hyperedges.forEach((he, idx) => {
        const memberNodes = he.nodeIds
          .map((id) => nodeMap.get(id))
          .filter((n): n is SimNode => n !== undefined && n.x !== undefined && n.y !== undefined);

        if (memberNodes.length < 2) return;

        const points: [number, number][] = memberNodes.map((n) => [n.x!, n.y!]);

        // Add padding points around each node for a nicer hull
        const padded: [number, number][] = [];
        const pad = 30;
        points.forEach(([px, py]) => {
          padded.push([px - pad, py - pad]);
          padded.push([px + pad, py - pad]);
          padded.push([px - pad, py + pad]);
          padded.push([px + pad, py + pad]);
        });

        const hull = convexHull(padded);
        if (hull.length < 3) return;

        const isSelected = selectedHyperedgeId === he.id;
        const fillColor = he.color || EDGE_PALETTE[idx % EDGE_PALETTE.length];
        const strokeColor = EDGE_STROKE_PALETTE[idx % EDGE_STROKE_PALETTE.length];

        const line = d3
          .line<[number, number]>()
          .x((d) => d[0])
          .y((d) => d[1])
          .curve(d3.curveCatmullRomClosed.alpha(0.5));

        hullGroup
          .append('path')
          .datum(hull)
          .attr('d', line)
          .attr('fill', isSelected ? fillColor.replace('0.15', '0.3') : fillColor)
          .attr('stroke', isSelected ? '#3b82f6' : strokeColor)
          .attr('stroke-width', isSelected ? 2 : 1)
          .attr('cursor', 'pointer')
          .on('click', (event: MouseEvent) => {
            event.stopPropagation();
            onSelectHyperedge(he.id);
          })
          .append('title')
          .text(`Hyperedge: ${he.id}\nType: ${he.type}\nNodes: ${he.nodeIds.length}\nWeight: ${he.weight.toFixed(3)}`);
      });
    }

    // Draw nodes
    const nodeGroup = mainGroup
      .append('g')
      .attr('class', 'nodes')
      .selectAll<SVGGElement, SimNode>('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('cursor', 'pointer')
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node circles
    nodeGroup
      .append('circle')
      .attr('r', 12)
      .attr('fill', (d) => getRiskColor(d.riskLevel))
      .attr('stroke', (d) => {
        if (cascadeState?.disruptedNodes.has(d.id)) return '#ef4444';
        if (selectedNodeId === d.id) return '#3b82f6';
        return 'rgba(255,255,255,0.2)';
      })
      .attr('stroke-width', (d) => {
        if (selectedNodeId === d.id || cascadeState?.disruptedNodes.has(d.id)) return 3;
        return 1.5;
      })
      .attr('filter', (d) => (selectedNodeId === d.id ? 'url(#glow)' : 'none'))
      .each(function (d) {
        if (cascadeState?.disruptedNodes.has(d.id)) {
          d3.select(this).attr('fill', '#ef4444').attr('stroke', '#fca5a5').attr('stroke-width', 3);
        }
        if (cascadeState?.atRiskNodes.has(d.id)) {
          d3.select(this).classed('at-risk-pulse', true).attr('stroke', '#f97316').attr('stroke-width', 2.5);
        }
      });

    // Node labels
    nodeGroup
      .append('text')
      .text((d) => d.label.length > 10 ? d.label.slice(0, 10) + '...' : d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', 28)
      .attr('fill', '#d1d5db')
      .attr('font-size', '10px')
      .attr('pointer-events', 'none');

    // Node tooltips
    nodeGroup
      .append('title')
      .text((d) => `${d.label}\nID: ${d.id}\nType: ${d.type}\nRisk: ${d.riskLevel ?? 'N/A'}${d.riskScore != null ? `\nScore: ${(d.riskScore * 100).toFixed(1)}%` : ''}`);

    // Click handler on nodes
    nodeGroup.on('click', (event: MouseEvent, d: SimNode) => {
      event.stopPropagation();
      onSelectNode(d.id);
    });

    // Click on background to deselect
    svg.on('click', () => {
      // Intentionally left for the parent to handle deselection if needed
    });

    // Simulation tick
    simulation.on('tick', () => {
      updateHulls();

      nodeGroup.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [graphData, dimensions, selectedNodeId, selectedHyperedgeId, cascadeState, onSelectNode, onSelectHyperedge]);

  const handleZoomIn = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(300)
      .call(zoomRef.current.scaleBy, 1.5);
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(300)
      .call(zoomRef.current.scaleBy, 0.67);
  }, []);

  const handleFitView = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current)
      .transition()
      .duration(500)
      .call(zoomRef.current.transform, d3.zoomIdentity);
  }, []);

  if (!graphData) {
    return (
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm flex items-center justify-center h-[600px]">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-gray-500 mx-auto mb-3 animate-spin" />
          <p className="text-gray-400 text-sm">Load a dataset to visualize the hypergraph.</p>
        </div>
      </div>
    );
  }

  if (graphData.nodes.length === 0) {
    return (
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm flex items-center justify-center h-[600px]">
        <p className="text-gray-400 text-sm">The loaded dataset contains no nodes.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700/50 bg-gray-900/50">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">
            {graphData.nodes.length} nodes / {graphData.hyperedges.length} hyperedges
          </span>
          {cascadeState && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
              Cascade Active
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleFitView}
            className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
            title="Fit view"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="w-full h-[600px] bg-black">
        <svg
          ref={svgRef}
          width={dimensions.width}
          height={dimensions.height}
          className="w-full h-full"
        />
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 px-4 py-2 border-t border-gray-700/50 bg-gray-900/50 text-xs">
        {Object.entries(RISK_COLORS).map(([level, color]) => (
          <div key={level} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="text-gray-400">{level}</span>
          </div>
        ))}
        {cascadeState && (
          <>
            <div className="w-px h-3 bg-gray-700" />
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500 ring-2 ring-red-500/50" />
              <span className="text-gray-400">Disrupted</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-orange-500 animate-pulse" />
              <span className="text-gray-400">At Risk</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
