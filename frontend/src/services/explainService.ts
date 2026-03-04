import axios, { AxiosInstance } from 'axios';
import type { ExplainRequest, ExplainResponse } from '../types';

const API: AxiosInstance = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const explainService = {
  /**
   * Request SHAP-based explanations for one or more nodes.
   *
   * @param request - The explanation request containing node IDs and
   *                  the prediction type to explain.
   * @returns An ExplainResponse containing per-node explanations with
   *          hyperedge attributions, feature attributions, and recommendations.
   */
  async explain(request: ExplainRequest): Promise<ExplainResponse> {
    try {
      const response = await API.post<ExplainResponse>('/explain', request);
      console.log(
        `[explainService] explain: received explanations for ${response.data.explanations.length} node(s)`,
      );
      return response.data;
    } catch (error) {
      console.error('[explainService] explain failed:', error);
      throw error;
    }
  },
};

export default explainService;
