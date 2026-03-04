import { useState, useCallback } from 'react';
import datasetService from '../services/datasetService';
import type {
  DatasetInfo,
  HypergraphData,
  GraphSummary,
  DatasetLoadRequest,
} from '../types';

export interface UseHypergraphReturn {
  /** All available datasets fetched from the backend */
  datasets: DatasetInfo[];
  /** The ID of the currently selected / loaded dataset */
  selectedDatasetId: string | null;
  /** Graph data (nodes + hyperedges) for the loaded dataset */
  graphData: HypergraphData | null;
  /** Summary statistics for the loaded graph */
  graphSummary: GraphSummary | null;
  /** ID of the currently selected node (for detail panels, etc.) */
  selectedNodeId: string | null;
  /** ID of the currently selected hyperedge */
  selectedHyperedgeId: string | null;
  /** True while any async operation is in flight */
  loading: boolean;
  /** Most recent error message, or null */
  error: string | null;

  /** Fetch the list of available datasets from the backend */
  fetchDatasets: () => Promise<void>;
  /** Load a dataset by ID, optionally with load-time configuration */
  loadDataset: (datasetId: string, options?: DatasetLoadRequest) => Promise<void>;
  /** Select a node by ID (pass null to deselect) */
  selectNode: (nodeId: string | null) => void;
  /** Select a hyperedge by ID (pass null to deselect) */
  selectHyperedge: (edgeId: string | null) => void;
  /** Directly set graph data (useful for uploaded / client-side data) */
  setGraphData: (data: HypergraphData) => void;
}

export function useHypergraph(): UseHypergraphReturn {
  // ── State ──────────────────────────────────────────────────────────────
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [graphData, setGraphDataState] = useState<HypergraphData | null>(null);
  const [graphSummary, setGraphSummary] = useState<GraphSummary | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedHyperedgeId, setSelectedHyperedgeId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // ── Methods ────────────────────────────────────────────────────────────

  const fetchDatasets = useCallback(async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const result = await datasetService.listDatasets();
      setDatasets(result);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Failed to fetch datasets';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDataset = useCallback(
    async (datasetId: string, options?: DatasetLoadRequest): Promise<void> => {
      setLoading(true);
      setError(null);
      try {
        const response = await datasetService.loadDataset(datasetId, options);
        setSelectedDatasetId(datasetId);
        setGraphSummary(response.graphSummary);
        // Reset node / edge selection when a new dataset is loaded
        setSelectedNodeId(null);
        setSelectedHyperedgeId(null);
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : 'Failed to load dataset';
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const selectNode = useCallback((nodeId: string | null): void => {
    setSelectedNodeId(nodeId);
  }, []);

  const selectHyperedge = useCallback((edgeId: string | null): void => {
    setSelectedHyperedgeId(edgeId);
  }, []);

  const setGraphData = useCallback((data: HypergraphData): void => {
    setGraphDataState(data);
    // Derive a quick summary from the data itself
    const avgSize =
      data.hyperedges.length > 0
        ? data.hyperedges.reduce((sum, e) => sum + e.nodeIds.length, 0) /
          data.hyperedges.length
        : 0;
    const maxPossibleEdges = data.nodes.length > 0 ? Math.pow(2, data.nodes.length) - 1 : 0;
    const density = maxPossibleEdges > 0 ? data.hyperedges.length / maxPossibleEdges : 0;

    setGraphSummary({
      nodes: data.nodes.length,
      hyperedges: data.hyperedges.length,
      avgHyperedgeSize: parseFloat(avgSize.toFixed(2)),
      density: parseFloat(density.toFixed(6)),
    });
    // Reset selections
    setSelectedNodeId(null);
    setSelectedHyperedgeId(null);
  }, []);

  // ── Return ─────────────────────────────────────────────────────────────

  return {
    datasets,
    selectedDatasetId,
    graphData,
    graphSummary,
    selectedNodeId,
    selectedHyperedgeId,
    loading,
    error,
    fetchDatasets,
    loadDataset,
    selectNode,
    selectHyperedge,
    setGraphData,
  };
}
