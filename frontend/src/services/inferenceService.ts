import axios, { AxiosInstance } from 'axios';
import type { PredictionInput, PredictionOutput, AnalysisResult, ModelInfo } from '../types';

class InferenceService {
  private api: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.api = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Get model information and metadata
   */
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await this.api.get<ModelInfo>('/model/info');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch model info:', error);
      throw error;
    }
  }

  /**
   * Run inference on supply chain data
   */
  async predict(input: PredictionInput): Promise<PredictionOutput> {
    try {
      const response = await this.api.post<PredictionOutput>('/predict', input);
      return response.data;
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  }

  /**
   * Analyze supply chain and get risk assessment
   */
  async analyze(input: PredictionInput): Promise<AnalysisResult> {
    try {
      const response = await this.api.post<AnalysisResult>('/analyze', input);
      return response.data;
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  }

  /**
   * Upload CSV file and get predictions
   */
  async uploadAndPredict(file: File): Promise<PredictionOutput> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await this.api.post<PredictionOutput>(
        '/upload/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Upload and predict failed:', error);
      throw error;
    }
  }

  /**
   * Get health status of the service
   */
  async getHealth(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await this.api.get<{ status: string; timestamp: string }>('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  /**
   * Get training history
   */
  async getTrainingHistory(): Promise<Record<string, number[]>> {
    try {
      const response = await this.api.get<Record<string, number[]>>('/training/history');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch training history:', error);
      throw error;
    }
  }
}

export default new InferenceService();
