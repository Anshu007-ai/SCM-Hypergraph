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
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50',
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
          <Upload className="w-12 h-12 text-gray-400 mb-2" />
          <p className="text-lg font-medium text-gray-700 mb-1">
            Drag and drop your CSV file
          </p>
          <p className="text-sm text-gray-500">
            or click to browse (max 10MB)
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-700 text-sm">Processing file...</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
          <AlertCircle className="w-5 h-5 text-red-600 mr-3 mt-0.5 flex-shrink-0" />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {success && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start">
          <CheckCircle className="w-5 h-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
          <p className="text-green-700 text-sm">{success}</p>
        </div>
      )}

      {result && (
        <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-3">Prediction Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-blue-50 rounded">
              <p className="text-sm text-gray-600">Price Predictions</p>
              <p className="text-xl font-bold text-blue-600">
                {result.pricePredictions.length}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded">
              <p className="text-sm text-gray-600">Change Forecasts</p>
              <p className="text-xl font-bold text-green-600">
                {result.changePredictions.length}
              </p>
            </div>
            <div className="p-3 bg-purple-50 rounded">
              <p className="text-sm text-gray-600">Criticality Scores</p>
              <p className="text-xl font-bold text-purple-600">
                {result.criticalityScores.length}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
