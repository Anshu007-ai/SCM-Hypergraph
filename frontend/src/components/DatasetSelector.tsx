import React from 'react';
import { Database, Loader2, CheckCircle, BarChart3, Network } from 'lucide-react';
import type { DatasetInfo, GraphSummary } from '../types';

interface DatasetSelectorProps {
  datasets: DatasetInfo[];
  selectedDataset: string | null;
  onSelect: (datasetId: string) => void;
  onLoad: (datasetId: string) => void;
  isLoading: boolean;
  graphSummary: GraphSummary | null;
}

export const DatasetSelector: React.FC<DatasetSelectorProps> = ({
  datasets,
  selectedDataset,
  onSelect,
  onLoad,
  isLoading,
  graphSummary,
}) => {
  if (!datasets || datasets.length === 0) {
    return (
      <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-8 text-center">
        <Database className="w-12 h-12 text-gray-500 mx-auto mb-3" />
        <p className="text-gray-400 text-sm">No datasets available.</p>
        <p className="text-gray-500 text-xs mt-1">
          Check your connection or try refreshing.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Dataset Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((dataset) => {
          const isSelected = selectedDataset === dataset.id;
          return (
            <button
              key={dataset.id}
              onClick={() => onSelect(dataset.id)}
              className={`text-left rounded-xl border p-5 transition-all duration-200 backdrop-blur-sm ${
                isSelected
                  ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500/40 shadow-lg shadow-blue-500/10'
                  : 'border-gray-700 bg-black/80 hover:border-gray-500 hover:bg-gray-900/80'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Database className={`w-4 h-4 ${isSelected ? 'text-blue-400' : 'text-gray-500'}`} />
                  <h3 className="font-semibold text-white text-sm truncate">
                    {dataset.name}
                  </h3>
                </div>
                {isSelected && (
                  <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0" />
                )}
              </div>

              <p className="text-gray-400 text-xs mb-4 line-clamp-2 leading-relaxed">
                {dataset.description}
              </p>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-1.5 text-gray-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span>
                    <span className="text-white font-medium">{(dataset.nodes ?? 0).toLocaleString()}</span> nodes
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-gray-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500" />
                  <span>
                    <span className="text-white font-medium">{(dataset.hyperedges ?? 0).toLocaleString()}</span> hyperedges
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-gray-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                  <span className="truncate">{dataset.timeSpan}</span>
                </div>
                <div className="flex items-center gap-1.5 text-gray-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                  <span className="truncate">{dataset.source}</span>
                </div>
              </div>

              {dataset.primaryTask && (
                <div className="mt-3 pt-3 border-t border-gray-700/50">
                  <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">
                    Task:
                  </span>
                  <span className="text-xs text-gray-300 ml-1.5">{dataset.primaryTask}</span>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Load Button */}
      <div className="flex items-center justify-between rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-4">
        <div className="text-sm text-gray-400">
          {selectedDataset
            ? `Selected: ${datasets.find((d) => d.id === selectedDataset)?.name ?? selectedDataset}`
            : 'Select a dataset above to load it'}
        </div>
        <button
          onClick={() => selectedDataset && onLoad(selectedDataset)}
          disabled={!selectedDataset || isLoading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/20"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading...
            </>
          ) : (
            <>
              <Database className="w-4 h-4" />
              Load Dataset
            </>
          )}
        </button>
      </div>

      {/* Graph Summary */}
      {graphSummary && (
        <div className="rounded-xl border border-gray-700 bg-black/80 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Graph Summary</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <SummaryStat
              label="Nodes"
              value={(graphSummary.nodes ?? 0).toLocaleString()}
              icon={<Network className="w-4 h-4 text-emerald-400" />}
            />
            <SummaryStat
              label="Hyperedges"
              value={(graphSummary.hyperedges ?? 0).toLocaleString()}
              icon={<Network className="w-4 h-4 text-purple-400" />}
            />
            <SummaryStat
              label="Avg Size"
              value={(graphSummary.avgHyperedgeSize ?? 0).toFixed(2)}
              icon={<BarChart3 className="w-4 h-4 text-amber-400" />}
            />
            <SummaryStat
              label="Density"
              value={(graphSummary.density ?? 0).toFixed(4)}
              icon={<BarChart3 className="w-4 h-4 text-cyan-400" />}
            />
          </div>
        </div>
      )}
    </div>
  );
};

const SummaryStat: React.FC<{
  label: string;
  value: string;
  icon: React.ReactNode;
}> = ({ label, value, icon }) => (
  <div className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3">
    <div className="flex items-center gap-2 mb-1">
      {icon}
      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">
        {label}
      </span>
    </div>
    <p className="text-xl font-bold text-white">{value}</p>
  </div>
);
