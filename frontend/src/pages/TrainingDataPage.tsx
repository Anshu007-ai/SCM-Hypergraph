import { useState } from 'react';
import { Download, FileSpreadsheet, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';

export default function TrainingDataPage({ onBack }: { onBack: () => void }) {
  const [isDownloading, setIsDownloading] = useState(false);

  const downloadAsHTML = () => {
    setIsDownloading(true);
    const html = generateHTMLContent();
    const blob = new Blob([html], { type: 'text/html' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Training_Data_Consolidation.html';
    a.click();
    window.URL.revokeObjectURL(url);
    setIsDownloading(false);
  };

  const downloadAsJSON = () => {
    setIsDownloading(true);
    const jsonData = {
      summary: {
        totalDataPoints: 9742955,
        totalRows: 196199,
        activeDatasets: 4,
        generatedDate: new Date().toISOString(),
      },
      datasets: [
        {
          name: 'DataCo Supply Chain',
          rows: 180519,
          columns: 53,
          size: '180 MB',
          timespan: '2015–2018',
          domain: 'E-commerce Logistics',
          description: 'E-commerce logistics data covering international shipping corridors',
          features: ['Order priority', 'Shipping mode', 'Delivery status', 'Product category', 'Profit ratio', 'Late delivery risk'],
          dataPoints: 9567507,
          percentage: 98.2,
        },
        {
          name: 'Maintenance (AI4I 2020)',
          rows: 10000,
          columns: 14,
          size: '2.5 MB',
          timespan: 'Synthetic (2020)',
          domain: 'Predictive Maintenance',
          description: 'Machine sensor and failure mode data (TWF, HDF, PWF, OSF, RNF)',
          features: ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Machine type'],
          dataPoints: 120000,
          percentage: 1.23,
        },
        {
          name: 'Automotive BOM',
          rows: 5566,
          columns: 6,
          size: '1.2 MB',
          timespan: '2020–2023',
          domain: 'Manufacturing',
          description: 'Supplier-component relationships in vehicle manufacturing',
          features: ['Component type', 'Supplier concentration', 'Lead time', 'Substitutability', 'Assembly level'],
          dataPoints: 36930,
          percentage: 0.38,
        },
        {
          name: 'IndiGo Aviation 2025',
          rows: 114,
          columns: 18,
          size: '50 KB',
          timespan: 'Jan–Dec 2025',
          domain: 'Aviation',
          description: 'IndiGo operational and disruption data (8 months: Jan, Apr, Jul-Dec)',
          features: ['Cancellation rate', 'OTP score', 'Fleet grounded %', 'Pilot ratio', 'MRO domestic %', 'Market share'],
          dataPoints: 18518,
          percentage: 0.19,
        },
      ],
      consolidationProcess: {
        step1: 'Normalization: All features scaled to [0,1] range using DataAdapter',
        step2: 'Graph Extraction: Build node-node relationships from raw data (supplier-component, order-corridor, etc.)',
        step3: 'Label Generation: Apply HCI formula with joint failure + engineering impact + propagation risk',
        step4: 'Tensorization: Convert to PyTorch tensors with incidence matrices and timestamps',
        step5: 'Training: 70% train / 15% val / 15% test split',
      },
      trainingConfig: {
        optimizer: 'Adam',
        learningRate: 0.001,
        batchSize: 32,
        epochs: 100,
        earlyStopping: true,
        patience: 10,
        scheduler: 'StepLR (step=30, gamma=0.5)',
      },
      performanceMetrics: {
        accuracy: 0.947,
        f1Score: 0.901,
        precision: 0.932,
        recall: 0.892,
        inferenceTime: '<60ms (CPU), ~12ms (GPU)',
        modelSize: '218K parameters',
      },
    };
    const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Training_Data_Consolidated.json';
    a.click();
    window.URL.revokeObjectURL(url);
    setIsDownloading(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-700 sticky top-0 z-50 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="p-2 hover:bg-slate-700 rounded-lg transition"
              aria-label="Go back"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-3xl font-bold">Training Data Consolidation</h1>
              <p className="text-blue-400 text-sm">Complete inventory of all datasets, sources, and features</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={downloadAsHTML}
              disabled={isDownloading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition disabled:opacity-50"
            >
              <Download className="w-4 h-4" />
              {isDownloading ? 'Downloading...' : 'HTML'}
            </button>
            <button
              onClick={downloadAsJSON}
              disabled={isDownloading}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg transition disabled:opacity-50"
            >
              <FileSpreadsheet className="w-4 h-4" />
              {isDownloading ? 'Downloading...' : 'JSON'}
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Summary Stats */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-12"
        >
          {[
            { label: 'Total Data Points', value: '9.7M', color: 'from-blue-500' },
            { label: 'Total Rows', value: '196K', color: 'from-emerald-500' },
            { label: 'Active Datasets', value: '4', color: 'from-orange-500' },
            { label: 'Features', value: '91', color: 'from-purple-500' },
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.15 + idx * 0.05 }}
              className={`bg-gradient-to-br ${stat.color} to-transparent rounded-lg p-6 border border-slate-700`}
            >
              <div className="text-sm text-slate-300 mb-2">{stat.label}</div>
              <div className="text-3xl font-bold">{stat.value}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Datasets Overview */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6 text-blue-400">Datasets Overview</h2>
          <div className="overflow-x-auto rounded-lg border border-slate-700">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-800 border-b border-slate-700">
                  <th className="px-6 py-3 text-left font-semibold">Dataset</th>
                  <th className="px-6 py-3 text-left font-semibold">Domain</th>
                  <th className="px-6 py-3 text-center font-semibold">Rows</th>
                  <th className="px-6 py-3 text-center font-semibold">Features</th>
                  <th className="px-6 py-3 text-center font-semibold">Size</th>
                  <th className="px-6 py-3 text-center font-semibold">Time Span</th>
                  <th className="px-6 py-3 text-center font-semibold">% Data</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: 'DataCo Supply Chain', domain: 'E-commerce Logistics', rows: '180,519', cols: '53', size: '180 MB', span: '2015–2018', pct: '98.2%' },
                  { name: 'Maintenance (AI4I)', domain: 'Predictive Maintenance', rows: '10,000', cols: '14', size: '2.5 MB', span: 'Synthetic', pct: '1.23%' },
                  { name: 'Automotive BOM', domain: 'Manufacturing', rows: '5,566', cols: '6', size: '1.2 MB', span: '2020–2023', pct: '0.38%' },
                  { name: 'IndiGo Aviation', domain: 'Aviation', rows: '114', cols: '18', size: '50 KB', span: 'Jan–Dec 2025', pct: '0.19%' },
                ].map((d, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-slate-800' : ''}>
                    <td className="px-6 py-3 font-medium text-blue-300">{d.name}</td>
                    <td className="px-6 py-3 text-slate-300">{d.domain}</td>
                    <td className="px-6 py-3 text-center">{d.rows}</td>
                    <td className="px-6 py-3 text-center">{d.cols}</td>
                    <td className="px-6 py-3 text-center text-slate-400">{d.size}</td>
                    <td className="px-6 py-3 text-center text-slate-400">{d.span}</td>
                    <td className="px-6 py-3 text-center font-semibold text-emerald-400">{d.pct}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* DataCo Details */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-4 text-blue-400">DataCo Supply Chain (180.5K rows)</h2>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold text-emerald-400 mb-2">Key Features</h3>
            <p className="text-slate-300 text-sm mb-4">
              Order ID, Customer ID, Shipping Mode (Standard, First Class, Same Day, Second Class), Product Category,
              Delivery Status (Delivered, Lost, Cancelled), Late Delivery Risk, Sales (USD), Product Price, Order
              Priority (H/M/L), Quantity Ordered, Profit Margin
            </p>
            <h3 className="font-semibold text-emerald-400 mb-2">Data Statistics</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Unique Orders</div>
                <div className="font-bold text-lg">180,519</div>
              </div>
              <div>
                <div className="text-slate-400">Time Span</div>
                <div className="font-bold text-lg">Jan 2015 – Jul 2018</div>
              </div>
              <div>
                <div className="text-slate-400">Shipping Modes</div>
                <div className="font-bold text-lg">4 types</div>
              </div>
              <div>
                <div className="text-slate-400">Late Delivery Risk</div>
                <div className="font-bold text-lg">0–1 continuous</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Maintenance Details */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-4 text-blue-400">Maintenance (10K rows)</h2>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold text-emerald-400 mb-2">Failure Modes</h3>
            <p className="text-slate-300 text-sm mb-4">
              Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF),
              Random Failure (RNF) — models 5 fault modes across 8 machine types in a synthetic production environment.
            </p>
            <h3 className="font-semibold text-emerald-400 mb-2">Sensor Features</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Air Temperature (K)</div>
                <div className="font-bold">295.3–309.5</div>
              </div>
              <div>
                <div className="text-slate-400">Process Temperature (K)</div>
                <div className="font-bold">305.7–313.8</div>
              </div>
              <div>
                <div className="text-slate-400">Rotational Speed (rpm)</div>
                <div className="font-bold">1168–2886</div>
              </div>
              <div>
                <div className="text-slate-400">Torque (Nm)</div>
                <div className="font-bold">3.8–76.0</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* BOM Details */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-4 text-blue-400">Automotive BOM (5.5K rows)</h2>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold text-emerald-400 mb-2">Component Data</h3>
            <p className="text-slate-300 text-sm mb-4">
              Bill-of-Materials with supplier-component relationships, lead times, substitutability scores, assembly levels,
              quality reject rates, and cost tier classification for automotive supply chains.
            </p>
            <h3 className="font-semibold text-emerald-400 mb-2">Structure</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Total Components</div>
                <div className="font-bold text-lg">5,566</div>
              </div>
              <div>
                <div className="text-slate-400">Assembly Levels</div>
                <div className="font-bold text-lg">1–7 (hierarchical)</div>
              </div>
              <div>
                <div className="text-slate-400">Suppliers</div>
                <div className="font-bold text-lg">847</div>
              </div>
              <div>
                <div className="text-slate-400">Substitutability</div>
                <div className="font-bold text-lg">0–1 continuous</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* IndiGo Details */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-4 text-blue-400">IndiGo Aviation 2025 (114 rows)</h2>
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold text-emerald-400 mb-2">Crisis Timeline</h3>
            <p className="text-slate-300 text-sm mb-4">
              Dec 2025 operational crisis triggered by FDTL Phase 2 regulatory shock and Pratt & Whitney engine supply chain
              failure. Includes 8 months of operational data (Jan, Apr, Jul–Dec 2025).
            </p>
            <h3 className="font-semibold text-emerald-400 mb-2">Key Metrics</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400">Nodes Modeled</div>
                <div className="font-bold text-lg">114 entities</div>
              </div>
              <div>
                <div className="text-slate-400">Cancellations</div>
                <div className="font-bold text-lg">4,500+</div>
              </div>
              <div>
                <div className="text-slate-400">Passengers Affected</div>
                <div className="font-bold text-lg">9.82L</div>
              </div>
              <div>
                <div className="text-slate-400">Fleet Grounded</div>
                <div className="font-bold text-lg">21%</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Data Consolidation Process */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6 text-blue-400">Data Consolidation Process</h2>
          <div className="space-y-4">
            {[
              { step: 1, name: 'Data Ingestion', desc: 'Load 4 datasets with DataAdapter unified normalization to [0,1] range' },
              { step: 2, name: 'Graph Construction', desc: 'Extract node-node relationships and build supply chain DAG' },
              { step: 3, name: 'Hyperedge Discovery', desc: 'Identify multi-way relationships via DynamicHyperedgeConstructor (temporal co-occurrence)' },
              { step: 4, name: 'Risk Label Generation', desc: 'Compute node criticality using HCI formula (joint failure + engineering impact + propagation)' },
              { step: 5, name: 'Tensorization', desc: 'Convert to PyTorch tensors: node features, incidence matrix, timestamps, edge types' },
              { step: 6, name: 'Train/Val/Test Split', desc: '70% train / 15% validation / 15% test (stratified by criticality class)' },
            ].map((item, idx) => (
              <div
                key={idx}
                className="flex gap-4 p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-blue-500 transition"
              >
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center font-bold">
                  {item.step}
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-emerald-400">{item.name}</div>
                  <p className="text-slate-300 text-sm">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Training Configuration */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6 text-blue-400">Training Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="font-semibold text-emerald-400 mb-4">Hyperparameters</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-slate-400">Optimizer</span><span>Adam</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Learning Rate</span><span>0.001</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Batch Size</span><span>32</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Epochs</span><span>100</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Early Stopping</span><span>Enabled (patience=10)</span></div>
                <div className="flex justify-between"><span className="text-slate-400">LR Scheduler</span><span>StepLR (step=30, γ=0.5)</span></div>
              </div>
            </div>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h3 className="font-semibold text-emerald-400 mb-4">Data Distribution</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-slate-400">Training Set</span><span className="text-blue-300 font-bold">70%</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Validation Set</span><span className="text-emerald-300 font-bold">15%</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Test Set</span><span className="text-orange-300 font-bold">15%</span></div>
                <div className="border-t border-slate-600 pt-2 mt-2"></div>
                <div className="flex justify-between"><span className="text-slate-400">Stratification</span><span>By criticality class</span></div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6 text-blue-400">Performance Metrics (HT-HGNN v2.0)</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { label: 'Accuracy', value: '94.7%', color: 'from-blue-500' },
              { label: 'F1 Score', value: '0.901', color: 'from-emerald-500' },
              { label: 'Precision', value: '0.932', color: 'from-purple-500' },
              { label: 'Recall', value: '0.892', color: 'from-orange-500' },
              { label: 'Inference Time', value: '<60ms', color: 'from-pink-500' },
              { label: 'Model Size', value: '218K Params', color: 'from-indigo-500' },
            ].map((metric, idx) => (
              <div
                key={idx}
                className={`bg-gradient-to-br ${metric.color} to-transparent rounded-lg p-6 border border-slate-700`}
              >
                <div className="text-sm text-slate-300 mb-2">{metric.label}</div>
                <div className="text-2xl font-bold">{metric.value}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* File Locations */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold mb-6 text-blue-400">File Locations</h2>
          <div className="space-y-4">
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-blue-300 break-all">Data set/DataCo/</div>
              <div className="text-xs text-slate-400 mt-1">DataCoSupplyChainDataset.csv (180.5K rows)</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-blue-300 break-all">Data set/Maintenance/</div>
              <div className="text-xs text-slate-400 mt-1">ai4i2020.csv (10K rows)</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-blue-300 break-all">Data set/BOM/</div>
              <div className="text-xs text-slate-400 mt-1">train_set.csv, test_set.csv (5.5K rows)</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-blue-300 break-all">Data set/IndiGo/</div>
              <div className="text-xs text-slate-400 mt-1">indigo_disruption_2025.csv (114 rows, 8 months)</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-emerald-300 break-all">outputs/datasets/</div>
              <div className="text-xs text-slate-400 mt-1">Processed & pickled data (*.pkl)</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <div className="font-mono text-sm text-emerald-300 break-all">outputs/checkpoints/</div>
              <div className="text-xs text-slate-400 mt-1">Trained model weights (best_all.pt, etc.)</div>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

function generateHTMLContent(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Training Data Consolidation - HT-HGNN v2.0</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; line-height: 1.6; color: #333; background: #f9fafb; }
    .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
    h1 { font-size: 2.5em; color: #1e40af; margin-bottom: 10px; page-break-after: avoid; }
    h2 { font-size: 1.8em; color: #1e40af; margin-top: 40px; margin-bottom: 20px; page-break-after: avoid; }
    h3 { font-size: 1.2em; color: #059669; margin-top: 20px; margin-bottom: 10px; }
    p { margin-bottom: 15px; text-align: justify; }
    .summary { background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-top: 20px; }
    .stat-box { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 6px; text-align: center; }
    .stat-value { font-size: 1.8em; font-weight: bold; }
    .stat-label { font-size: 0.9em; opacity: 0.9; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; page-break-inside: avoid; }
    th { background: #1e40af; color: white; padding: 12px; text-align: left; font-weight: 600; }
    td { padding: 12px; border-bottom: 1px solid #e5e7eb; }
    tr:nth-child(even) { background: #f3f4f6; }
    .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #1e40af; page-break-inside: avoid; }
    .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
    .data-box { background: #f3f4f6; padding: 15px; border-radius: 6px; border-left: 3px solid #059669; }
    .data-box strong { color: #059669; }
    .file-path { background: #f0f4f8; padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; color: #0369a1; margin: 10px 0; }
    .page-break { page-break-before: always; }
    @media print {
      body { background: white; }
      .container { max-width: 100%; padding: 20px; }
      h1 { page-break-after: avoid; }
      table { page-break-inside: avoid; }
      .section { page-break-inside: avoid; }
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Cover -->
    <div style="text-align: center; padding: 60px 0; page-break-after: always;">
      <h1>Training Data Consolidation</h1>
      <p style="color: #3b82f6; font-size: 1.2em; margin-bottom: 20px;">HT-HGNN v2.0 Supply Chain Risk Analysis</p>
      <p style="color: #666; font-size: 0.95em;">Complete inventory of all datasets, sources, features, and training configuration</p>
      <p style="color: #999; margin-top: 40px; font-size: 0.85em;">Generated on ${new Date().toLocaleDateString()}</p>
    </div>

    <!-- Summary -->
    <div class="summary">
      <h2 style="color: white; margin-top: 0;">Project Summary</h2>
      <div class="summary-grid">
        <div class="stat-box">
          <div class="stat-value">9.7M</div>
          <div class="stat-label">Data Points</div>
        </div>
        <div class="stat-box">
          <div class="stat-value">196K</div>
          <div class="stat-label">Total Rows</div>
        </div>
        <div class="stat-box">
          <div class="stat-value">4</div>
          <div class="stat-label">Datasets</div>
        </div>
        <div class="stat-box">
          <div class="stat-value">91</div>
          <div class="stat-label">Features</div>
        </div>
      </div>
    </div>

    <!-- Datasets Overview -->
    <h2>Datasets Overview</h2>
    <table>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Domain</th>
          <th>Rows</th>
          <th>Features</th>
          <th>Size</th>
          <th>Time Span</th>
          <th>% Data</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>DataCo Supply Chain</strong></td>
          <td>E-commerce Logistics</td>
          <td>180,519</td>
          <td>53</td>
          <td>180 MB</td>
          <td>2015–2018</td>
          <td><strong>98.2%</strong></td>
        </tr>
        <tr>
          <td><strong>Maintenance (AI4I)</strong></td>
          <td>Predictive Maintenance</td>
          <td>10,000</td>
          <td>14</td>
          <td>2.5 MB</td>
          <td>Synthetic</td>
          <td><strong>1.23%</strong></td>
        </tr>
        <tr>
          <td><strong>Automotive BOM</strong></td>
          <td>Manufacturing</td>
          <td>5,566</td>
          <td>6</td>
          <td>1.2 MB</td>
          <td>2020–2023</td>
          <td><strong>0.38%</strong></td>
        </tr>
        <tr>
          <td><strong>IndiGo Aviation</strong></td>
          <td>Aviation</td>
          <td>114</td>
          <td>18</td>
          <td>50 KB</td>
          <td>Jan–Dec 2025</td>
          <td><strong>0.19%</strong></td>
        </tr>
      </tbody>
    </table>

    <div class="section">
      <h3>DataCo Supply Chain (180.5K rows)</h3>
      <p><strong>Domain:</strong> E-commerce Logistics | <strong>Time Span:</strong> 2015–2018 | <strong>Size:</strong> 180 MB</p>
      <p><strong>Description:</strong> The DataCo Global Supply Chain dataset contains detailed e-commerce logistics data covering 180K+ orders across international shipping corridors. It captures multi-modal shipping (Standard, First Class, Same Day, Second Class), delivery performance, product profitability, and late-delivery risk factors.</p>
      <p><strong>Key Features:</strong> Order ID, Customer ID, Shipping Mode, Product Category, Delivery Status, Late Delivery Risk, Sales (USD), Product Price, Order Priority, Quantity Ordered, Profit Margin</p>
      <p><strong>Data Points:</strong> 9,567,507 (98.2% of total)</p>
    </div>

    <div class="section">
      <h3>Maintenance (AI4I 2020) (10K rows)</h3>
      <p><strong>Domain:</strong> Predictive Maintenance | <strong>Source:</strong> UCI ML Repository | <strong>Size:</strong> 2.5 MB</p>
      <p><strong>Description:</strong> Machine sensor and failure mode data modeling 5 distinct failure types: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failure (RNF). Captures correlated failures across thermal zones and production lines.</p>
      <p><strong>Sensor Features:</strong> Air Temperature (295–310 K), Process Temperature (306–314 K), Rotational Speed (1168–2886 rpm), Torque (3.8–76 Nm), Tool Wear (0–255)</p>
      <p><strong>Data Points:</strong> 120,000 (1.23% of total)</p>
    </div>

    <div class="section page-break">
      <h3>Automotive BOM (5.5K rows)</h3>
      <p><strong>Domain:</strong> Manufacturing | <strong>Time Span:</strong> 2020–2023 | <strong>Size:</strong> 1.2 MB</p>
      <p><strong>Description:</strong> Automotive Bill-of-Materials modeling supplier-component relationships in vehicle manufacturing. Each node represents a physical component with supply chain attributes. Hyperedges represent multi-component assemblies where single component failure disrupts entire assembly.</p>
      <p><strong>Key Features:</strong> Component Type, Supplier Concentration, Lead Time, Substitutability, Assembly Level, Quality Reject Rate, Cost Tier, Geographic Origin</p>
      <p><strong>Data Points:</strong> 36,930 (0.38% of total)</p>
    </div>

    <div class="section">
      <h3>IndiGo Aviation 2025 (114 rows)</h3>
      <p><strong>Domain:</strong> Aviation Operations | <strong>Crisis Period:</strong> Dec 2025 | <strong>Data Span:</strong> Jan, Apr, Jul–Dec 2025</p>
      <p><strong>Description:</strong> IndiGo operational crisis triggered by FDTL Phase 2 regulatory shock and Pratt & Whitney engine supply chain failure. Models full cascade: FDTL Shock → Pilot Shortage → Route Cancellations → Passenger Displacement → Railway Surge → Market Redistribution.</p>
      <p><strong>Key Metrics:</strong> Cancellation Rate, OTP Score, Fleet Grounded %, Pilot-per-Aircraft Ratio, MRO Domestic %, Market Share, Regulatory Compliance, Demand Volatility</p>
      <p><strong>Crisis Impact:</strong> 4,500+ cancellations, 9.82L passengers affected, 21% fleet grounded, OTP drop from 84% to 62%</p>
      <p><strong>Data Points:</strong> 18,518 (0.19% of total)</p>
    </div>

    <h2>Data Consolidation Process</h2>
    <div class="grid-2">
      <div class="data-box">
        <strong>Step 1: Data Ingestion</strong><br>Load 4 datasets with DataAdapter unified normalization to [0,1] range
      </div>
      <div class="data-box">
        <strong>Step 2: Graph Construction</strong><br>Extract node-node relationships and build supply chain DAG
      </div>
      <div class="data-box">
        <strong>Step 3: Hyperedge Discovery</strong><br>Identify multi-way relationships via DynamicHyperedgeConstructor (temporal co-occurrence)
      </div>
      <div class="data-box">
        <strong>Step 4: Risk Labels</strong><br>Compute node criticality using HCI formula (joint failure + engineering impact + propagation)
      </div>
      <div class="data-box">
        <strong>Step 5: Tensorization</strong><br>Convert to PyTorch tensors: node features, incidence matrix, timestamps, edge types
      </div>
      <div class="data-box">
        <strong>Step 6: Train/Val/Test</strong><br>70% train / 15% val / 15% test split (stratified by criticality class)
      </div>
    </div>

    <h2>Training Configuration</h2>
    <table>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Optimizer</td>
          <td>Adam</td>
        </tr>
        <tr>
          <td>Learning Rate</td>
          <td>0.001</td>
        </tr>
        <tr>
          <td>Batch Size</td>
          <td>32</td>
        </tr>
        <tr>
          <td>Epochs</td>
          <td>100</td>
        </tr>
        <tr>
          <td>Early Stopping</td>
          <td>Enabled (patience=10)</td>
        </tr>
        <tr>
          <td>LR Scheduler</td>
          <td>StepLR (step=30, γ=0.5)</td>
        </tr>
        <tr>
          <td>Train / Val / Test Split</td>
          <td>70% / 15% / 15%</td>
        </tr>
        <tr>
          <td>Stratification</td>
          <td>By criticality class</td>
        </tr>
      </tbody>
    </table>

    <h2>Performance Metrics (HT-HGNN v2.0)</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Accuracy</td>
          <td><strong>94.7%</strong></td>
        </tr>
        <tr>
          <td>F1 Score (Macro)</td>
          <td><strong>0.901</strong></td>
        </tr>
        <tr>
          <td>Precision</td>
          <td><strong>0.932</strong></td>
        </tr>
        <tr>
          <td>Recall</td>
          <td><strong>0.892</strong></td>
        </tr>
        <tr>
          <td>Inference Time (CPU)</td>
          <td><strong>&lt;60ms</strong> per batch</td>
        </tr>
        <tr>
          <td>Inference Time (GPU)</td>
          <td><strong>~12ms</strong> per batch</td>
        </tr>
        <tr>
          <td>Model Size</td>
          <td><strong>218K parameters</strong></td>
        </tr>
      </tbody>
    </table>

    <h2>File Locations</h2>
    <p><strong>Raw Data:</strong></p>
    <div class="file-path">Data set/DataCo/DataCoSupplyChainDataset.csv (180.5K rows)</div>
    <div class="file-path">Data set/Maintenance/ai4i2020.csv (10K rows)</div>
    <div class="file-path">Data set/BOM/train_set.csv, test_set.csv (5.5K rows)</div>
    <div class="file-path">Data set/IndiGo/indigo_disruption_2025.csv (114 rows)</div>
    
    <p><strong>Processed Data:</strong></p>
    <div class="file-path">outputs/datasets/*.pkl (Pickled tensors)</div>
    <div class="file-path">outputs/datasets/hypergraph.json (Graph structure, 55 MB)</div>
    
    <p><strong>Models & Checkpoints:</strong></p>
    <div class="file-path">outputs/checkpoints/best_all.pt (Best model weights)</div>
    <div class="file-path">outputs/training_history.json (Training metrics, 6 MB)</div>
  </div>
</body>
</html>`;
}
