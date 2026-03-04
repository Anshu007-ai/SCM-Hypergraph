import { useState, useCallback } from 'react';
import explainService from '../services/explainService';
import type { ExplainRequest, NodeExplanation } from '../types';

export interface UseExplainReturn {
  /** Map of nodeId to its explanation */
  explanations: Map<string, NodeExplanation>;
  /** True while an explanation request is in flight */
  isExplaining: boolean;
  /** Most recent error message, or null */
  error: string | null;
  /** The node whose explanation is currently shown in the UI */
  selectedNodeId: string | null;
  /** Convenience getter: the explanation for the selected node, or null */
  selectedExplanation: NodeExplanation | null;

  /** Request explanations for multiple nodes at once */
  explainNodes: (
    nodeIds: string[],
    predictionType: ExplainRequest['predictionType'],
  ) => Promise<void>;
  /** Shortcut to request an explanation for a single node */
  explainNode: (
    nodeId: string,
    predictionType: ExplainRequest['predictionType'],
  ) => Promise<void>;
  /** Set which node's explanation to display */
  selectNode: (nodeId: string | null) => void;
  /** Clear all explanations and reset state */
  clearExplanations: () => void;
}

export function useExplain(): UseExplainReturn {
  // ── State ──────────────────────────────────────────────────────────────
  const [explanations, setExplanations] = useState<Map<string, NodeExplanation>>(
    () => new Map(),
  );
  const [isExplaining, setIsExplaining] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // ── Derived ────────────────────────────────────────────────────────────

  const selectedExplanation: NodeExplanation | null =
    selectedNodeId !== null ? (explanations.get(selectedNodeId) ?? null) : null;

  // ── Methods ────────────────────────────────────────────────────────────

  const explainNodes = useCallback(
    async (
      nodeIds: string[],
      predictionType: ExplainRequest['predictionType'],
    ): Promise<void> => {
      if (nodeIds.length === 0) return;

      setIsExplaining(true);
      setError(null);

      try {
        const request: ExplainRequest = { nodeIds, predictionType };
        const response = await explainService.explain(request);

        setExplanations((prev) => {
          const next = new Map(prev);
          for (const explanation of response.explanations) {
            next.set(explanation.nodeId, explanation);
          }
          return next;
        });

        // Auto-select the first node if nothing is currently selected
        if (selectedNodeId === null && response.explanations.length > 0) {
          setSelectedNodeId(response.explanations[0].nodeId);
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : 'Failed to generate explanations';
        setError(message);
      } finally {
        setIsExplaining(false);
      }
    },
    [selectedNodeId],
  );

  const explainNode = useCallback(
    async (
      nodeId: string,
      predictionType: ExplainRequest['predictionType'],
    ): Promise<void> => {
      setIsExplaining(true);
      setError(null);

      try {
        const request: ExplainRequest = {
          nodeIds: [nodeId],
          predictionType,
        };
        const response = await explainService.explain(request);

        setExplanations((prev) => {
          const next = new Map(prev);
          for (const explanation of response.explanations) {
            next.set(explanation.nodeId, explanation);
          }
          return next;
        });

        // Auto-select the explained node
        setSelectedNodeId(nodeId);
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : 'Failed to generate explanation';
        setError(message);
      } finally {
        setIsExplaining(false);
      }
    },
    [],
  );

  const selectNode = useCallback((nodeId: string | null): void => {
    setSelectedNodeId(nodeId);
  }, []);

  const clearExplanations = useCallback((): void => {
    setExplanations(new Map());
    setError(null);
    setSelectedNodeId(null);
  }, []);

  // ── Return ─────────────────────────────────────────────────────────────

  return {
    explanations,
    isExplaining,
    error,
    selectedNodeId,
    selectedExplanation,
    explainNodes,
    explainNode,
    selectNode,
    clearExplanations,
  };
}
