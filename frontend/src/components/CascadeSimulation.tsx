import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';
import { Play, RotateCcw, AlertTriangle, Network } from 'lucide-react';

/* -----------------------------------------------------------------------
   TYPES
   ----------------------------------------------------------------------- */

type NodeKind = 'aircraft' | 'sector' | 'airport' | 'route' | 'crew' | 'regulatory';

interface CascadeNode {
  id: string;
  label: string;
  kind: NodeKind;
  cx: number;  // SVG x
  cy: number;  // SVG y
}

interface Hyperedge {
  id: string;
  label: string;
  members: string[];
  color: string;
}

type CascadeState = 'idle' | 'hop0' | 'hop1' | 'hop2' | 'hop3';

/* -----------------------------------------------------------------------
   DATA — 12 nodes, 4 hyperedges
   ----------------------------------------------------------------------- */

const NODES: CascadeNode[] = [
  // Aircraft (blue circles)
  { id: 'ac1', label: 'Flight 6E-302', kind: 'aircraft', cx: 180, cy: 90 },
  { id: 'ac2', label: 'Flight 6E-417', kind: 'aircraft', cx: 380, cy: 60 },
  { id: 'ac3', label: 'Flight 6E-891', kind: 'aircraft', cx: 580, cy: 100 },
  // Sectors (orange squares)
  { id: 'sec1', label: 'Sector W-12', kind: 'sector', cx: 120, cy: 230 },
  { id: 'sec2', label: 'Sector E-07', kind: 'sector', cx: 520, cy: 220 },
  // Airports (green diamonds)
  { id: 'apt1', label: 'DEL (Delhi)', kind: 'airport', cx: 280, cy: 190 },
  { id: 'apt2', label: 'BOM (Mumbai)', kind: 'airport', cx: 460, cy: 190 },
  // Routes (purple circles)
  { id: 'rt1', label: 'DEL→BOM Route', kind: 'route', cx: 360, cy: 300 },
  { id: 'rt2', label: 'BOM→BLR Route', kind: 'route', cx: 540, cy: 320 },
  // Crew pools (yellow circles)
  { id: 'crew1', label: 'DEL Crew Pool', kind: 'crew', cx: 160, cy: 350 },
  { id: 'crew2', label: 'BOM Crew Pool', kind: 'crew', cx: 440, cy: 380 },
  // Regulatory
  { id: 'reg', label: 'DGCA FDTL Cap', kind: 'regulatory', cx: 300, cy: 420 },
];

const HYPEREDGES: Hyperedge[] = [
  { id: 'he1', label: 'Delhi Hub Operations', members: ['ac1', 'sec1', 'apt1', 'crew1'], color: 'rgba(59,130,246,0.08)' },
  { id: 'he2', label: 'Mumbai Hub Operations', members: ['ac2', 'apt2', 'sec2', 'crew2'], color: 'rgba(251,146,60,0.08)' },
  { id: 'he3', label: 'DEL-BOM Corridor', members: ['ac1', 'ac2', 'apt1', 'apt2', 'rt1'], color: 'rgba(16,185,129,0.08)' },
  { id: 'he4', label: 'FDTL Regulatory Chain', members: ['crew1', 'crew2', 'reg', 'rt1', 'rt2'], color: 'rgba(239,68,68,0.08)' },
];

/** Links derived from hyperedge co-membership */
function deriveLinks(): { from: string; to: string }[] {
  const linkSet = new Set<string>();
  const links: { from: string; to: string }[] = [];
  for (const he of HYPEREDGES) {
    for (let i = 0; i < he.members.length; i++) {
      for (let j = i + 1; j < he.members.length; j++) {
        const key = [he.members[i], he.members[j]].sort().join('-');
        if (!linkSet.has(key)) {
          linkSet.add(key);
          links.push({ from: he.members[i], to: he.members[j] });
        }
      }
    }
  }
  return links;
}

const LINKS = deriveLinks();

/** Propagation table: for each seed node, which nodes light up at each hop. */
const PROPAGATION: Record<string, string[][]> = {
  ac1:  [['ac1'], ['sec1', 'apt1', 'crew1'], ['ac2', 'rt1', 'apt2'], ['crew2', 'sec2', 'reg']],
  ac2:  [['ac2'], ['apt2', 'sec2', 'crew2'], ['ac1', 'rt1', 'apt1'], ['crew1', 'sec1', 'reg']],
  ac3:  [['ac3'], ['sec2', 'apt2'], ['ac2', 'crew2', 'rt2'], ['rt1', 'reg']],
  sec1: [['sec1'], ['ac1', 'apt1', 'crew1'], ['ac2', 'rt1'], ['apt2', 'reg']],
  sec2: [['sec2'], ['ac2', 'apt2', 'crew2'], ['ac3', 'rt1', 'rt2'], ['ac1', 'reg']],
  apt1: [['apt1'], ['ac1', 'ac2', 'sec1', 'rt1'], ['crew1', 'apt2', 'crew2'], ['reg', 'rt2']],
  apt2: [['apt2'], ['ac2', 'sec2', 'rt1', 'crew2'], ['ac1', 'apt1', 'rt2'], ['crew1', 'reg']],
  rt1:  [['rt1'], ['ac1', 'ac2', 'apt1', 'apt2', 'crew1'], ['sec1', 'sec2', 'crew2'], ['ac3', 'reg']],
  rt2:  [['rt2'], ['crew1', 'crew2', 'reg'], ['rt1', 'apt1', 'apt2'], ['ac1', 'ac2']],
  crew1:[['crew1'], ['ac1', 'sec1', 'apt1', 'reg'], ['ac2', 'rt1', 'crew2'], ['apt2', 'rt2']],
  crew2:[['crew2'], ['ac2', 'apt2', 'sec2', 'reg'], ['ac1', 'rt1', 'rt2'], ['apt1', 'crew1']],
  reg:  [['reg'], ['crew1', 'crew2', 'rt1', 'rt2'], ['ac1', 'ac2', 'apt1', 'apt2'], ['sec1', 'sec2', 'ac3']],
};

/* -----------------------------------------------------------------------
   NODE RENDERING HELPERS
   ----------------------------------------------------------------------- */

const KIND_CONFIG: Record<NodeKind, { fill: string; stroke: string; shape: 'circle' | 'square' | 'diamond' | 'hexagon' }> = {
  aircraft:   { fill: '#3B82F6', stroke: '#60A5FA', shape: 'circle' },
  sector:     { fill: '#F97316', stroke: '#FB923C', shape: 'square' },
  airport:    { fill: '#10B981', stroke: '#34D399', shape: 'diamond' },
  route:      { fill: '#8B5CF6', stroke: '#A78BFA', shape: 'circle' },
  crew:       { fill: '#EAB308', stroke: '#FACC15', shape: 'circle' },
  regulatory: { fill: '#EF4444', stroke: '#F87171', shape: 'hexagon' },
};

function NodeShape({ node, state }: { node: CascadeNode; state: 'normal' | 'seed' | 'affected' }) {
  const cfg = KIND_CONFIG[node.kind];
  const r = 18;
  const fillColor = state === 'seed' ? '#EF4444' : state === 'affected' ? '#F59E0B' : cfg.fill;
  const strokeColor = state === 'seed' ? '#FCA5A5' : state === 'affected' ? '#FDE68A' : cfg.stroke;
  const pulseClass = state !== 'normal' ? 'animate-pulse' : '';

  const baseProps = {
    className: pulseClass,
    fill: fillColor,
    stroke: strokeColor,
    strokeWidth: state !== 'normal' ? 2.5 : 1.5,
    fillOpacity: state !== 'normal' ? 0.9 : 0.6,
  };

  switch (cfg.shape) {
    case 'square':
      return <rect x={node.cx - r * 0.75} y={node.cy - r * 0.75} width={r * 1.5} height={r * 1.5} rx={3} {...baseProps} />;
    case 'diamond': {
      const pts = `${node.cx},${node.cy - r} ${node.cx + r},${node.cy} ${node.cx},${node.cy + r} ${node.cx - r},${node.cy}`;
      return <polygon points={pts} {...baseProps} />;
    }
    case 'hexagon': {
      const hex = Array.from({ length: 6 }, (_, i) => {
        const angle = (Math.PI / 3) * i - Math.PI / 6;
        return `${node.cx + r * Math.cos(angle)},${node.cy + r * Math.sin(angle)}`;
      }).join(' ');
      return <polygon points={hex} {...baseProps} />;
    }
    default:
      return <circle cx={node.cx} cy={node.cy} r={r} {...baseProps} />;
  }
}

/* -----------------------------------------------------------------------
   HYPEREDGE POLYGON
   ----------------------------------------------------------------------- */

function HyperedgePolygon({ he }: { he: Hyperedge }) {
  const memberNodes = he.members.map(mid => NODES.find(n => n.id === mid)!).filter(Boolean);
  if (memberNodes.length < 3) return null;
  // Convex hull (simple: sort by angle from centroid)
  const cx = memberNodes.reduce((s, n) => s + n.cx, 0) / memberNodes.length;
  const cy = memberNodes.reduce((s, n) => s + n.cy, 0) / memberNodes.length;
  const sorted = [...memberNodes].sort(
    (a, b) => Math.atan2(a.cy - cy, a.cx - cx) - Math.atan2(b.cy - cy, b.cx - cx),
  );
  const pad = 32;
  const pts = sorted.map(n => {
    const dx = n.cx - cx;
    const dy = n.cy - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const scale = dist > 0 ? (dist + pad) / dist : 1;
    return `${cx + dx * scale},${cy + dy * scale}`;
  }).join(' ');
  return <polygon points={pts} fill={he.color} stroke="rgba(255,255,255,0.04)" strokeWidth={1} />;
}

/* -----------------------------------------------------------------------
   COMPONENT
   ----------------------------------------------------------------------- */

const CascadeSimulation: React.FC = () => {
  const [seed, setSeed] = useState<string>('ac1');
  const [cascadeState, setCascadeState] = useState<CascadeState>('idle');
  const [affectedNodes, setAffectedNodes] = useState<Set<string>>(new Set());
  const [currentHop, setCurrentHop] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const sectionRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: '-80px' });

  const reset = useCallback(() => {
    timerRef.current.forEach(clearTimeout);
    timerRef.current = [];
    setCascadeState('idle');
    setAffectedNodes(new Set());
    setCurrentHop(0);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => timerRef.current.forEach(clearTimeout), []);

  const simulate = useCallback(() => {
    reset();
    const path = PROPAGATION[seed];
    if (!path) return;

    // TODO: Replace with POST /simulate/cascade API call
    // const response = await fetch(`${API_BASE}/simulate/cascade`, {
    //   method: 'POST',
    //   body: JSON.stringify({ seed_node: seed, max_hops: 3 }),
    // });

    const hops: CascadeState[] = ['hop0', 'hop1', 'hop2', 'hop3'];
    const accumulated = new Set<string>();

    for (let hop = 0; hop < Math.min(path.length, 4); hop++) {
      const t = setTimeout(() => {
        path[hop].forEach(id => accumulated.add(id));
        setAffectedNodes(new Set(accumulated));
        setCascadeState(hops[hop]);
        setCurrentHop(hop);
      }, hop * 800);
      timerRef.current.push(t);
    }
  }, [seed, reset]);

  const getNodeState = (nodeId: string): 'normal' | 'seed' | 'affected' => {
    if (cascadeState === 'idle') return 'normal';
    const path = PROPAGATION[seed];
    if (path && path[0]?.includes(nodeId)) return 'seed';
    if (affectedNodes.has(nodeId)) return 'affected';
    return 'normal';
  };

  return (
    <div ref={sectionRef}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.7 }}
        className="glass rounded-2xl border border-white/5 overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-5 pb-3">
          <div>
            <h3 className="text-white font-semibold text-sm flex items-center gap-2">
              <Network className="w-4 h-4 text-accent-blue" />
              Live Cascade Simulation
            </h3>
            <p className="text-gray-500 text-xs mt-0.5">
              Pick a seed node and watch disruption propagate through hyperedge connections
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={seed}
              onChange={e => { reset(); setSeed(e.target.value); }}
              className="bg-navy-800 border border-white/10 rounded-lg text-xs text-gray-300 px-3 py-1.5 outline-none focus:border-accent-blue/50"
            >
              {NODES.map(n => (
                <option key={n.id} value={n.id}>{n.label}</option>
              ))}
            </select>
            <button
              onClick={simulate}
              disabled={cascadeState !== 'idle'}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-blue/20 text-accent-blue text-xs font-semibold hover:bg-accent-blue/30 transition-colors disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5" /> Simulate
            </button>
            <button
              onClick={reset}
              className="p-1.5 rounded-lg bg-white/5 text-gray-400 hover:bg-white/10 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* SVG Diagram */}
        <div className="px-5 pb-2">
          <svg viewBox="0 0 700 470" className="w-full" style={{ maxHeight: 420 }}>
            {/* Hyperedge polygons */}
            {HYPEREDGES.map(he => <HyperedgePolygon key={he.id} he={he} />)}

            {/* Links */}
            {LINKS.map(({ from, to }, i) => {
              const a = NODES.find(n => n.id === from)!;
              const b = NODES.find(n => n.id === to)!;
              const active = affectedNodes.has(from) && affectedNodes.has(to);
              return (
                <line
                  key={i}
                  x1={a.cx} y1={a.cy} x2={b.cx} y2={b.cy}
                  stroke={active ? 'rgba(245,158,11,0.4)' : 'rgba(255,255,255,0.07)'}
                  strokeWidth={active ? 1.5 : 0.8}
                />
              );
            })}

            {/* Nodes */}
            {NODES.map(node => (
              <g key={node.id}>
                <NodeShape node={node} state={getNodeState(node.id)} />
                <text
                  x={node.cx} y={node.cy + 28}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.65)"
                  fontSize={9}
                  fontFamily="sans-serif"
                >
                  {node.label}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* Stats bar */}
        <div className="px-5 pb-5">
          <AnimatePresence mode="wait">
            <motion.div
              key={cascadeState}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="flex items-center gap-6 py-3 px-4 rounded-xl bg-navy-800/60 border border-white/5"
            >
              <div className="flex items-center gap-2">
                <AlertTriangle className={`w-4 h-4 ${cascadeState !== 'idle' ? 'text-amber-400' : 'text-gray-600'}`} />
                <span className="text-xs text-gray-400">
                  Nodes affected: <span className={`font-bold ${cascadeState !== 'idle' ? 'text-amber-400' : 'text-gray-500'}`}>{affectedNodes.size} / 12</span>
                </span>
              </div>
              <div className="text-xs text-gray-400">
                Cascade depth: <span className={`font-bold ${cascadeState !== 'idle' ? 'text-accent-blue' : 'text-gray-500'}`}>{cascadeState === 'idle' ? '—' : `${currentHop} hops`}</span>
              </div>
              {cascadeState !== 'idle' && (
                <div className="ml-auto">
                  <div className="flex gap-1">
                    {[0, 1, 2, 3].map(h => (
                      <div
                        key={h}
                        className={`w-8 h-1.5 rounded-full transition-colors duration-300 ${
                          h <= currentHop ? 'bg-amber-400' : 'bg-white/10'
                        }`}
                      />
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Legend */}
        <div className="px-5 pb-5 flex flex-wrap gap-x-5 gap-y-1.5 text-[10px] text-gray-500">
          {([
            ['aircraft', 'Aircraft', '●'],
            ['sector', 'Sector', '■'],
            ['airport', 'Airport', '◆'],
            ['route', 'Route', '●'],
            ['crew', 'Crew Pool', '●'],
            ['regulatory', 'Regulatory', '⬡'],
          ] as const).map(([kind, label, icon]) => (
            <span key={kind} className="flex items-center gap-1">
              <span style={{ color: KIND_CONFIG[kind].fill }}>{icon}</span> {label}
            </span>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default CascadeSimulation;
