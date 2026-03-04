import React, { useEffect } from 'react';
import { Database, Network, BarChart3, Cpu, Upload, Loader2 } from 'lucide-react';
import { DatasetSelector } from '../components/DatasetSelector';
import { FileUpload } from '../components/FileUpload';
import type {
  DatasetInfo,
  GraphSummary,
  ModelInfo,
  PredictionOutput,
} from '../types';

interface DashboardPageProps {
  /** Available datasets from the backend */
  datasets: DatasetInfo[];
  /** Currently selected dataset ID */
  selectedDatasetId: string | null;
  /** Callback when a dataset card is clicked */
  onSelectDataset: (datasetId: string) => void;
  /** Callback when the "Load Dataset" button is clicked */
  onDatasetLoad: (datasetId: string) => void;
  /** Callback when a CSV file is uploaded */
  onFileUpload: (file: File) => Promise<PredictionOutput>;
  /** True while a dataset is being loaded or a file is being processed */
  isLoading: boolean;
  /** Summary statistics for the currently loaded graph */
  graphSummary: GraphSummary | null;
  /** Model information fetched from the backend */
  modelInfo: ModelInfo | null;
  /** Fetch the list of available datasets */
  onFetchDatasets: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  datasets,
  selectedDatasetId,
  onSelectDataset,
  onDatasetLoad,
  onFileUpload,
  isLoading,
  graphSummary,
  modelInfo,
  onFetchDatasets,
}) => {
  // Fetch available datasets on mount
  useEffect(() => {
    onFetchDatasets();
  }, [onFetchDatasets]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-12 space-y-10">
      {/* Page Title */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-gray-400 text-sm">
          Load a dataset or upload a CSV to get started with HT-HGNN analysis.
        </p>
      </div>

      {/* Quick Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <QuickStatCard
          icon={<Network className="w-5 h-5 text-emerald-400" />}
          label="Total Nodes"
          value={graphSummary?.nodes != null ? graphSummary.nodes.toLocaleString() : '--'}
        />
        <QuickStatCard
          icon={<BarChart3 className="w-5 h-5 text-purple-400" />}
          label="Hyperedges"
          value={graphSummary?.hyperedges != null ? graphSummary.hyperedges.toLocaleString() : '--'}
        />
        <QuickStatCard
          icon={<Database className="w-5 h-5 text-amber-400" />}
          label="Avg Risk"
          value={
            graphSummary?.density != null
              ? graphSummary.density < 0.01
                ? graphSummary.density.toExponential(2)
                : graphSummary.density.toFixed(4)
              : '--'
          }
        />
        <QuickStatCard
          icon={<Cpu className="w-5 h-5 text-cyan-400" />}
          label="Model"
          value={modelInfo?.name ?? '--'}
          subValue={modelInfo?.parameters != null ? `${modelInfo.parameters.toLocaleString()} params` : undefined}
        />
      </div>

      {/* Two-Column Layout: Dataset Selector + File Upload */}
      <div className="grid lg:grid-cols-3 gap-8">
        {/* Dataset Selector (takes 2 cols) */}
        <div className="lg:col-span-2">
          <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <Database className="w-5 h-5 text-blue-400" />
              <h2 className="text-lg font-semibold text-white">Select a Dataset</h2>
              {isLoading && (
                <Loader2 className="w-4 h-4 text-blue-400 animate-spin ml-auto" />
              )}
            </div>
            <DatasetSelector
              datasets={datasets}
              selectedDataset={selectedDatasetId}
              onSelect={onSelectDataset}
              onLoad={onDatasetLoad}
              isLoading={isLoading}
              graphSummary={graphSummary}
            />
          </div>
        </div>

        {/* File Upload + Model Info */}
        <div className="space-y-6">
          {/* File Upload */}
          <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-5">
              <Upload className="w-5 h-5 text-green-400" />
              <h2 className="text-lg font-semibold text-white">Upload CSV</h2>
            </div>
            <FileUpload onUpload={onFileUpload} isLoading={isLoading} />
          </div>

          {/* Model Information */}
          {modelInfo && (
            <div className="bg-black/80 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-5">
                <Cpu className="w-5 h-5 text-cyan-400" />
                <h2 className="text-lg font-semibold text-white">Model Info</h2>
              </div>
              <div className="space-y-3">
                <ModelInfoRow label="Name" value={modelInfo.name ?? '--'} />
                <ModelInfoRow label="Version" value={modelInfo.version ?? '--'} />
                <ModelInfoRow
                  label="Parameters"
                  value={modelInfo.parameters != null ? modelInfo.parameters.toLocaleString() : '--'}
                />
                <ModelInfoRow label="Device" value={modelInfo.device ?? '--'} />
                <ModelInfoRow label="Architecture" value={modelInfo.architecture ?? '--'} />
                <ModelInfoRow
                  label="Training Date"
                  value={modelInfo.trainingDate ? new Date(modelInfo.trainingDate).toLocaleDateString() : '--'}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/*  Helper sub-components                                                      */
/* -------------------------------------------------------------------------- */

const QuickStatCard: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
}> = ({ icon, label, value, subValue }) => (
  <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-4">
    <div className="flex items-center gap-2 mb-2">
      {icon}
      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">
        {label}
      </span>
    </div>
    <p className="text-2xl font-bold text-white truncate">{value}</p>
    {subValue && (
      <p className="text-xs text-gray-500 mt-1 truncate">{subValue}</p>
    )}
  </div>
);

const ModelInfoRow: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div>
    <p className="text-xs text-gray-500 font-medium">{label}</p>
    <p className="text-sm text-white font-semibold truncate">{value}</p>
  </div>
);
