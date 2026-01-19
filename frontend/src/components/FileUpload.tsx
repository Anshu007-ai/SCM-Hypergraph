import React, { useState } from 'react';
import clsx from 'clsx';
import { Upload, AlertCircle, CheckCircle } from 'lucide-react';
import type { PredictionOutput } from '../types';

interface FileUploadProps {
  onUpload: (file: File) => Promise<PredictionOutput>;
  isLoading?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onUpload, isLoading = false }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionOutput | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      await processFile(files[0]);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      await processFile(files[0]);
    }
  };

  const processFile = async (file: File) => {
    setError(null);
    setSuccess(null);
    setResult(null);

    // Validate file
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    try {
      const predictions = await onUpload(file);
      setResult(predictions);
      setSuccess(`Successfully analyzed ${file.name}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to process file'
      );
    }
  };

  return (
    <div className="w-full">
      <div
        className={clsx(
          'relative border-2 border-dashed rounded-lg p-8 transition-colors',
          isDragging ? 'border-blue-400 bg-blue-950/30' : 'border-gray-600 bg-gray-900/30',
          isLoading && 'opacity-50 pointer-events-none'
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          className="absolute inset-0 opacity-0 cursor-pointer"
          disabled={isLoading}
        />
        
        <div className="flex flex-col items-center justify-center pointer-events-none">
          <Upload className="w-12 h-12 text-gray-500 mb-2" />
          <p className="text-lg font-medium text-gray-300 mb-1">
            Drag and drop your CSV file
          </p>
          <p className="text-sm text-gray-500">
            or click to browse (max 10MB)
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="mt-4 p-4 bg-blue-950/30 border border-blue-700/50 rounded-lg">
          <p className="text-blue-400 text-sm">Processing file...</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-950/30 border border-red-700/50 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-400 mr-3 mt-0.5 flex-shrink-0" />
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {success && (
        <div className="mt-4 p-4 bg-green-950/30 border border-green-700/50 rounded-lg flex items-start">
          <CheckCircle className="w-5 h-5 text-green-400 mr-3 mt-0.5 flex-shrink-0" />
          <p className="text-green-400 text-sm">{success}</p>
        </div>
      )}

      {result && (
        <div className="mt-4 p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
          <h3 className="font-semibold text-white mb-3">Prediction Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-blue-950/50 border border-blue-700/30 rounded">
              <p className="text-sm text-gray-400">Price Predictions</p>
              <p className="text-xl font-bold text-blue-400">
                {result.pricePredictions.length}
              </p>
            </div>
            <div className="p-3 bg-green-950/50 border border-green-700/30 rounded">
              <p className="text-sm text-gray-400">Change Forecasts</p>
              <p className="text-xl font-bold text-green-400">
                {result.changePredictions.length}
              </p>
            </div>
            <div className="p-3 bg-purple-950/50 border border-purple-700/30 rounded">
              <p className="text-sm text-gray-400">Criticality Scores</p>
              <p className="text-xl font-bold text-purple-400">
                {result.criticalityScores.length}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
