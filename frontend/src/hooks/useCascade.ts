import { useState, useCallback, useRef } from 'react';
import simulationService from '../services/simulationService';
import type { CascadeRequest, CascadeResult, CascadeStep } from '../types';

/** Default interval (ms) used by the step-through animation */
const ANIMATION_INTERVAL_MS = 800;

export interface UseCascadeReturn {
  /** Full result returned by the backend after simulation completes */
  cascadeResult: CascadeResult | null;
  /** True while the simulation request is in flight / polling */
  isSimulating: boolean;
  /** Index of the currently displayed cascade step (0-based) */
  currentStep: number;
  /** The CascadeStep object for the current step index, or null */
  currentStepData: CascadeStep | null;
  /** Total number of steps in the timeline */
  totalSteps: number;
  /** Most recent error message, or null */
  error: string | null;
  /** Node IDs selected as initial shock sources */
  selectedShockNodes: string[];
  /** Shock magnitude (0 to 1) */
  shockMagnitude: number;
  /** Whether the step animation is currently playing */
  isPlaying: boolean;

  /** Submit a cascade simulation request and poll until completion */
  startSimulation: (request: CascadeRequest) => Promise<void>;
  /** Clear all simulation state */
  resetSimulation: () => void;
  /** Add a node ID to the shock-source selection */
  addShockNode: (nodeId: string) => void;
  /** Remove a node ID from the shock-source selection */
  removeShockNode: (nodeId: string) => void;
  /** Set the shock magnitude (clamped to [0, 1]) */
  setShockMagnitude: (magnitude: number) => void;
  /** Advance the timeline by one step */
  stepForward: () => void;
  /** Go back one step in the timeline */
  stepBackward: () => void;
  /** Start auto-advancing through the timeline */
  playAnimation: () => void;
  /** Stop auto-advancing */
  pauseAnimation: () => void;
}

export function useCascade(): UseCascadeReturn {
  // ── State ──────────────────────────────────────────────────────────────
  const [cascadeResult, setCascadeResult] = useState<CascadeResult | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedShockNodes, setSelectedShockNodes] = useState<string[]>([]);
  const [shockMagnitude, setShockMagnitudeState] = useState<number>(0.5);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  /** Interval handle for the auto-play animation */
  const animationRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Derived ────────────────────────────────────────────────────────────

  const timeline = cascadeResult?.timeline ?? [];
  const totalSteps = timeline.length;
  const currentStepData: CascadeStep | null =
    totalSteps > 0 && currentStep >= 0 && currentStep < totalSteps
      ? timeline[currentStep]
      : null;

  // ── Internal helpers ───────────────────────────────────────────────────

  const clearAnimation = useCallback((): void => {
    if (animationRef.current !== null) {
      clearInterval(animationRef.current);
      animationRef.current = null;
    }
    setIsPlaying(false);
  }, []);

  // ── Public methods ─────────────────────────────────────────────────────

  const startSimulation = useCallback(
    async (request: CascadeRequest): Promise<void> => {
      clearAnimation();
      setIsSimulating(true);
      setError(null);
      setCascadeResult(null);
      setCurrentStep(0);

      try {
        const { taskId } = await simulationService.runCascade(request);
        const result = await simulationService.pollUntilComplete(taskId);
        setCascadeResult(result);
        setCurrentStep(0);
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : 'Cascade simulation failed';
        setError(message);
      } finally {
        setIsSimulating(false);
      }
    },
    [clearAnimation],
  );

  const resetSimulation = useCallback((): void => {
    clearAnimation();
    setCascadeResult(null);
    setIsSimulating(false);
    setCurrentStep(0);
    setError(null);
    setSelectedShockNodes([]);
    setShockMagnitudeState(0.5);
  }, [clearAnimation]);

  const addShockNode = useCallback((nodeId: string): void => {
    setSelectedShockNodes((prev) =>
      prev.includes(nodeId) ? prev : [...prev, nodeId],
    );
  }, []);

  const removeShockNode = useCallback((nodeId: string): void => {
    setSelectedShockNodes((prev) => prev.filter((id) => id !== nodeId));
  }, []);

  const setShockMagnitude = useCallback((magnitude: number): void => {
    setShockMagnitudeState(Math.max(0, Math.min(1, magnitude)));
  }, []);

  const stepForward = useCallback((): void => {
    setCascadeResult((result) => {
      const max = (result?.timeline.length ?? 1) - 1;
      setCurrentStep((prev) => Math.min(prev + 1, max));
      return result;
    });
  }, []);

  const stepBackward = useCallback((): void => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const playAnimation = useCallback((): void => {
    // Avoid duplicate intervals
    if (animationRef.current !== null) return;

    setIsPlaying(true);
    animationRef.current = setInterval(() => {
      setCascadeResult((result) => {
        const max = (result?.timeline.length ?? 1) - 1;
        setCurrentStep((prev) => {
          if (prev >= max) {
            // Reached the end -- stop the animation
            clearAnimation();
            return prev;
          }
          return prev + 1;
        });
        return result;
      });
    }, ANIMATION_INTERVAL_MS);
  }, [clearAnimation]);

  const pauseAnimation = useCallback((): void => {
    clearAnimation();
  }, [clearAnimation]);

  // ── Return ─────────────────────────────────────────────────────────────

  return {
    cascadeResult,
    isSimulating,
    currentStep,
    currentStepData,
    totalSteps,
    error,
    selectedShockNodes,
    shockMagnitude,
    isPlaying,
    startSimulation,
    resetSimulation,
    addShockNode,
    removeShockNode,
    setShockMagnitude,
    stepForward,
    stepBackward,
    playAnimation,
    pauseAnimation,
  };
}
