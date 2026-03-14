import { useState, useMemo, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceDot,
} from 'recharts';
import { Target, Sliders, Info } from 'lucide-react';

/* -----------------------------------------------------------------------
   PARETO FRONT DATA GENERATION
   ----------------------------------------------------------------------- */

interface ParetoPoint {
  id: number;
  delay: number;         // objective 1: total delay minimisation (0–100, normalised)
  cascade: number;       // objective 2: cascade spread probability  (0–1)
  fairness: number;      // objective 3: resource fairness            (0–1, higher = fairer)
  conflicts: number;     // objective 4: conflict count               (integer)
  isKnee: boolean;
}

/** Generate 40 non-dominated Pareto-optimal points on a convex front. */
function generateParetoFront(): ParetoPoint[] {
  const points: ParetoPoint[] = [];
  for (let i = 0; i < 40; i++) {
    const t = i / 39; // parameter along the front
    // Convex trade-off: as delay reduction ↑, cascade risk ↑
    const delay = 15 + 75 * (1 - t) + (Math.random() - 0.5) * 6;
    const cascade = 0.08 + 0.82 * t * t + (Math.random() - 0.5) * 0.04;
    // Fairness inversely correlated with extreme solutions
    const fairness = 0.3 + 0.55 * Math.sin(Math.PI * t) + (Math.random() - 0.5) * 0.08;
    // Conflicts grow with aggressive delay-reduction
    const conflicts = Math.round(2 + 18 * (1 - t) + Math.random() * 4);
    points.push({
      id: i,
      delay: Math.max(5, Math.min(100, delay)),
      cascade: Math.max(0.05, Math.min(0.95, cascade)),
      fairness: Math.max(0.15, Math.min(0.95, fairness)),
      conflicts: Math.max(1, Math.min(24, conflicts)),
      isKnee: false,
    });
  }
  // Mark the knee point: best balanced trade-off (closest to utopia normalised)
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const normDelay = (100 - p.delay) / 100;     // higher is better
    const normCascade = (1 - p.cascade);           // lower cascade is better
    const normFairness = p.fairness;
    const normConflict = 1 - p.conflicts / 24;
    const dist = Math.sqrt(
      (1 - normDelay) ** 2 + (1 - normCascade) ** 2 +
      (1 - normFairness) ** 2 + (1 - normConflict) ** 2,
    );
    if (dist < bestDist) { bestDist = dist; bestIdx = i; }
  }
  points[bestIdx].isKnee = true;
  return points;
}

// Stable synthetic data
const PARETO_DATA = generateParetoFront();

/* -----------------------------------------------------------------------
   HELPERS
   ----------------------------------------------------------------------- */

function fairnessToColor(f: number): string {
  // green (fair) → red (unfair)
  const r = Math.round(240 * (1 - f) + 34 * f);
  const g = Math.round(60 * (1 - f) + 197 * f);
  const b = Math.round(60 * (1 - f) + 94 * f);
  return `rgb(${r},${g},${b})`;
}

function conflictsToSize(c: number): number {
  return 60 + (c / 24) * 300; // 60–360
}

/** Given weights, find the operating point that best matches the preference. */
function findOperatingPoint(
  points: ParetoPoint[],
  w1: number, w2: number, w3: number, w4: number,
): ParetoPoint {
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const score =
      w1 * ((100 - p.delay) / 100) +
      w2 * (1 - p.conflicts / 24) +
      w3 * p.fairness +
      w4 * (1 - p.cascade);
    if (score > bestScore) { bestScore = score; bestIdx = i; }
  }
  return points[bestIdx];
}

/* -----------------------------------------------------------------------
   CUSTOM TOOLTIP
   ----------------------------------------------------------------------- */

interface ParetoTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ParetoPoint }>;
}

const ParetoTooltip: React.FC<ParetoTooltipProps> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-navy-800 border border-white/10 rounded-xl px-4 py-3 text-xs shadow-2xl">
      <p className="text-white font-semibold mb-1">Solution #{p.id + 1}{p.isKnee ? ' ★ Knee Point' : ''}</p>
      <p className="text-gray-400">Delay Reduction: <span className="text-accent-blue font-bold">{p.delay.toFixed(1)}%</span></p>
      <p className="text-gray-400">Cascade Probability: <span className="text-amber-400 font-bold">{(p.cascade * 100).toFixed(1)}%</span></p>
      <p className="text-gray-400">Fairness: <span className="font-bold" style={{ color: fairnessToColor(p.fairness) }}>{p.fairness.toFixed(2)}</span></p>
      <p className="text-gray-400">Conflicts: <span className="text-red-400 font-bold">{p.conflicts}</span></p>
    </div>
  );
};

/* -----------------------------------------------------------------------
   COMPONENT
   ----------------------------------------------------------------------- */

const ParetoVisualiser: React.FC = () => {
  // Weights that sum to 1.0
  const [weights, setWeights] = useState({ w1: 0.35, w2: 0.20, w3: 0.20, w4: 0.25 });

  const sectionRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: '-80px' });

  const operatingPoint = useMemo(
    () => findOperatingPoint(PARETO_DATA, weights.w1, weights.w2, weights.w3, weights.w4),
    [weights],
  );

  const kneePoint = useMemo(() => PARETO_DATA.find(p => p.isKnee)!, []);

  /** Update one weight and re-normalise others proportionally. */
  const updateWeight = (key: 'w1' | 'w2' | 'w3' | 'w4', value: number) => {
    const clamped = Math.max(0, Math.min(1, value));
    const remaining = 1 - clamped;
    const others = (['w1', 'w2', 'w3', 'w4'] as const).filter(k => k !== key);
    const otherSum = others.reduce((s, k) => s + weights[k], 0);
    const next = { ...weights, [key]: clamped };
    if (otherSum > 0) {
      for (const k of others) next[k] = weights[k] * (remaining / otherSum);
    } else {
      for (const k of others) next[k] = remaining / 3;
    }
    setWeights(next);
  };

  const sliders: { key: 'w1' | 'w2' | 'w3' | 'w4'; label: string; color: string }[] = [
    { key: 'w1', label: 'Delay Minimisation', color: '#3B82F6' },
    { key: 'w2', label: 'Conflict Reduction', color: '#EF4444' },
    { key: 'w3', label: 'Resource Fairness', color: '#10B981' },
    { key: 'w4', label: 'Cascade Prevention', color: '#F59E0B' },
  ];

  return (
    <section id="pareto" className="relative py-24 md:py-32" ref={sectionRef}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-4">
            MOO Pareto Analysis
          </h2>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto mb-4">
            Multi-Objective Optimisation for ATC Disruption Response
          </p>
          <div className="h-1 w-20 mx-auto bg-gradient-to-r from-accent-blue to-accent-cyan rounded-full" />
        </motion.div>

        {/* Explanation */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="glass rounded-2xl p-5 border border-accent-blue/10 mb-8 flex items-start gap-3"
        >
          <Info className="w-5 h-5 text-accent-blue shrink-0 mt-0.5" />
          <p className="text-gray-300 text-sm leading-relaxed">
            A <strong className="text-white">Pareto front</strong> shows every solution where improving one objective necessarily worsens another —
            no solution on this curve is strictly dominated.
            For ATC decision-makers this means choosing the best trade-off between delay reduction, cascade containment,
            resource fairness, and conflict avoidance without hidden costs.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Chart — 2 cols */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="lg:col-span-2 glass rounded-2xl p-5 border border-white/5"
          >
            <h3 className="text-white font-semibold text-sm mb-1 flex items-center gap-2">
              <Target className="w-4 h-4 text-accent-blue" />
              4-Objective Pareto Front
            </h3>
            <p className="text-gray-500 text-xs mb-4">
              Colour = fairness (green → fair, red → unfair) · Size = conflict count · ★ = knee point
            </p>

            <ResponsiveContainer width="100%" height={420}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  type="number" dataKey="delay" name="Delay Reduction"
                  domain={[0, 100]} tickCount={6}
                  tick={{ fill: '#94A3B8', fontSize: 11 }}
                  label={{ value: 'Total Delay Minimisation (%)', position: 'bottom', offset: 2, style: { fill: '#94A3B8', fontSize: 11 } }}
                  stroke="rgba(255,255,255,0.1)"
                />
                <YAxis
                  type="number" dataKey="cascade" name="Cascade Prob."
                  domain={[0, 1]} tickCount={6}
                  tick={{ fill: '#94A3B8', fontSize: 11 }}
                  label={{ value: 'Cascade Spread Probability', angle: -90, position: 'insideLeft', offset: 6, style: { fill: '#94A3B8', fontSize: 11 } }}
                  stroke="rgba(255,255,255,0.1)"
                />
                <Tooltip content={<ParetoTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.15)' }} />
                <Scatter data={PARETO_DATA} isAnimationActive={false}>
                  {PARETO_DATA.map((p) => (
                    <Cell
                      key={p.id}
                      fill={fairnessToColor(p.fairness)}
                      fillOpacity={0.85}
                      r={Math.sqrt(conflictsToSize(p.conflicts) / Math.PI)}
                      stroke={p.isKnee ? '#FFFFFF' : 'transparent'}
                      strokeWidth={p.isKnee ? 2 : 0}
                    />
                  ))}
                </Scatter>
                {/* Knee point star marker */}
                <ReferenceDot
                  x={kneePoint.delay} y={kneePoint.cascade}
                  r={0} isFront
                  label={{
                    value: '★ Knee',
                    position: 'top',
                    style: { fill: '#FBBF24', fontSize: 12, fontWeight: 700 },
                    offset: 14,
                  }}
                />
                {/* Operating point marker */}
                <ReferenceDot
                  x={operatingPoint.delay} y={operatingPoint.cascade}
                  r={8} fill="#3B82F6" stroke="#FFFFFF" strokeWidth={2}
                  isFront
                  label={{
                    value: '● Selected',
                    position: 'bottom',
                    style: { fill: '#3B82F6', fontSize: 10, fontWeight: 600 },
                    offset: 12,
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Sliders + Summary — 1 col */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.35 }}
            className="space-y-5"
          >
            {/* Sliders */}
            <div className="glass rounded-2xl p-5 border border-white/5">
              <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-accent-cyan" />
                Objective Weights
              </h3>
              <div className="space-y-4">
                {sliders.map(({ key, label, color }) => (
                  <div key={key}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">{label}</span>
                      <span className="font-bold" style={{ color }}>{(weights[key] * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range" min="0" max="100" step="1"
                      value={Math.round(weights[key] * 100)}
                      onChange={e => updateWeight(key, parseInt(e.target.value) / 100)}
                      className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, ${color} ${weights[key] * 100}%, rgba(255,255,255,0.1) ${weights[key] * 100}%)`,
                        accentColor: color,
                      }}
                    />
                  </div>
                ))}
                <p className="text-[10px] text-gray-600 text-right">
                  Σ = {((weights.w1 + weights.w2 + weights.w3 + weights.w4) * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            {/* Summary Card */}
            <div className="glass rounded-2xl p-5 border border-accent-blue/15">
              <h3 className="text-white font-semibold text-sm mb-3">Selected Operating Point</h3>
              <div className="space-y-2 text-xs">
                {[
                  { label: 'Expected Delay Reduction', value: `${operatingPoint.delay.toFixed(1)}%`, color: '#3B82F6' },
                  { label: 'Conflict Probability', value: `${operatingPoint.conflicts} conflicts`, color: '#EF4444' },
                  { label: 'Fairness Score', value: operatingPoint.fairness.toFixed(3), color: '#10B981' },
                  { label: 'Cascade Risk', value: `${(operatingPoint.cascade * 100).toFixed(1)}%`, color: '#F59E0B' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
                    <span className="text-gray-400">{label}</span>
                    <span className="font-bold" style={{ color }}>{value}</span>
                  </div>
                ))}
              </div>
              {operatingPoint.isKnee && (
                <div className="mt-3 text-[10px] text-amber-400 bg-amber-500/10 rounded-lg px-3 py-1.5 text-center">
                  ★ This is the knee-point solution — the most balanced trade-off.
                </div>
              )}
            </div>

            {/* TODO */}
            <p className="text-[9px] text-gray-600 text-center">
              {/* TODO: Replace synthetic data with POST /moo/pareto API call */}
              Demo data — 40 synthetic non-dominated solutions
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default ParetoVisualiser;
