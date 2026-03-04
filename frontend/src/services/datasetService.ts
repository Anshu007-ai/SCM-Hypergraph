import axios, { AxiosInstance } from 'axios';
import type { DatasetInfo, DatasetLoadRequest, GraphSummary } from '../types';

const API: AxiosInstance = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/* ------------------------------------------------------------------
 * Helper: map snake_case backend JSON → camelCase frontend types
 * ----------------------------------------------------------------*/

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapDatasetInfo(raw: any): DatasetInfo {
  return {
    id: raw.id,
    name: raw.name,
    nodes: raw.nodes ?? raw.node_count ?? 0,
    hyperedges: raw.hyperedges ?? raw.hyperedge_count ?? 0,
    timeSpan: raw.timeSpan ?? raw.time_span ?? '',
    primaryTask: raw.primaryTask ?? raw.primary_task ?? '',
    source: raw.source ?? '',
    description: raw.description ?? '',
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapGraphSummary(raw: any): GraphSummary {
  return {
    nodes: raw.nodes ?? 0,
    hyperedges: raw.hyperedges ?? 0,
    avgHyperedgeSize: raw.avgHyperedgeSize ?? raw.avg_hyperedge_size ?? 0,
    density: raw.density ?? 0,
  };
}

const datasetService = {
  /**
   * Fetch the list of all available datasets from the backend.
   * @returns Array of DatasetInfo objects describing each dataset.
   */
  async listDatasets(): Promise<DatasetInfo[]> {
    try {
      const response = await API.get('/datasets');
      const datasets: DatasetInfo[] = (response.data as unknown[]).map(mapDatasetInfo);
      console.log(`[datasetService] listDatasets: received ${datasets.length} datasets`);
      return datasets;
    } catch (error) {
      console.error('[datasetService] listDatasets failed:', error);
      throw error;
    }
  },

  /**
   * Load a specific dataset into memory on the backend, optionally
   * applying temporal-window, minimum-hyperedge-size, or dynamic-edge filters.
   *
   * @param datasetId - The unique identifier of the dataset to load.
   * @param options   - Optional load-time configuration.
   * @returns An object containing the load status and a GraphSummary.
   */
  async loadDataset(
    datasetId: string,
    options?: DatasetLoadRequest,
  ): Promise<{ status: string; graphSummary: GraphSummary }> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const response = await API.post<any>(
        `/datasets/${encodeURIComponent(datasetId)}/load`,
        options ?? {},
      );
      const raw = response.data;
      const graphSummary = mapGraphSummary(raw.graphSummary ?? raw.graph_summary ?? {});
      const status: string = raw.status ?? 'loaded';
      console.log(
        `[datasetService] loadDataset(${datasetId}): status=${status}, ` +
        `nodes=${graphSummary.nodes}, hyperedges=${graphSummary.hyperedges}`,
      );
      return { status, graphSummary };
    } catch (error) {
      console.error(`[datasetService] loadDataset(${datasetId}) failed:`, error);
      throw error;
    }
  },
};

export default datasetService;
