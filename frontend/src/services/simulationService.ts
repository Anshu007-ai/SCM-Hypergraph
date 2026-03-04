import axios, { AxiosInstance } from 'axios';
import type { CascadeRequest, CascadeResult } from '../types';

const API: AxiosInstance = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/** Default polling interval in milliseconds */
const DEFAULT_POLL_INTERVAL_MS = 2000;

/** Maximum number of polling iterations before giving up */
const MAX_POLL_ITERATIONS = 300;

const simulationService = {
  /**
   * Submit a cascade simulation request.
   * The backend returns immediately with a task ID that can be polled.
   *
   * @param request - The cascade simulation parameters.
   * @returns An object with the taskId, initial status, and a pollUrl.
   */
  async runCascade(
    request: CascadeRequest,
  ): Promise<{ taskId: string; status: string; pollUrl: string }> {
    try {
      const response = await API.post<{ taskId: string; status: string; pollUrl: string }>(
        '/simulate/cascade',
        request,
      );
      console.log(
        `[simulationService] runCascade: taskId=${response.data.taskId}, status=${response.data.status}`,
      );
      return response.data;
    } catch (error) {
      console.error('[simulationService] runCascade failed:', error);
      throw error;
    }
  },

  /**
   * Retrieve the current status and (partial or complete) result for a task.
   *
   * @param taskId - The task identifier returned by runCascade.
   * @returns The latest CascadeResult for the given task.
   */
  async getTaskStatus(taskId: string): Promise<CascadeResult> {
    try {
      const response = await API.get<CascadeResult>(
        `/tasks/${encodeURIComponent(taskId)}`,
      );
      return response.data;
    } catch (error) {
      console.error(`[simulationService] getTaskStatus(${taskId}) failed:`, error);
      throw error;
    }
  },

  /**
   * Poll getTaskStatus until the task reaches a terminal state
   * (COMPLETED or FAILED).
   *
   * @param taskId     - The task identifier to poll.
   * @param intervalMs - Milliseconds between polls (default 2000).
   * @returns The final CascadeResult once the task is done.
   * @throws If the task fails or polling exceeds the maximum iterations.
   */
  async pollUntilComplete(
    taskId: string,
    intervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  ): Promise<CascadeResult> {
    let iterations = 0;

    while (iterations < MAX_POLL_ITERATIONS) {
      const result = await this.getTaskStatus(taskId);

      console.log(
        `[simulationService] pollUntilComplete(${taskId}): ` +
        `iteration=${iterations + 1}, status=${result.status}`,
      );

      if (result.status === 'COMPLETED') {
        return result;
      }

      if (result.status === 'FAILED') {
        throw new Error(
          `[simulationService] Task ${taskId} failed. Result: ${JSON.stringify(result)}`,
        );
      }

      // Wait before the next poll
      await new Promise<void>((resolve) => setTimeout(resolve, intervalMs));
      iterations++;
    }

    throw new Error(
      `[simulationService] pollUntilComplete(${taskId}): exceeded maximum ` +
      `${MAX_POLL_ITERATIONS} polling iterations (interval=${intervalMs}ms)`,
    );
  },
};

export default simulationService;
