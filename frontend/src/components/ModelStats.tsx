import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, Zap, Database, Cpu } from 'lucide-react';
import type { TrainingMetrics } from '../types';

interface ModelStatsProps {
  metrics: TrainingMetrics;
  device: string;
}

export const ModelStats: React.FC<ModelStatsProps> = ({ metrics, device }) => {
  const trainingTime = (metrics.totalTime / 60).toFixed(2);
  const improvementPercent = ((metrics.improvement / metrics.initialLoss) * 100).toFixed(2);

  const data = [
    { name: 'Initial', loss: metrics.initialLoss },
    { name: 'Final', loss: metrics.finalLoss },
  ];

  const stats = [
    { label: 'Initial Loss', value: metrics.initialLoss.toFixed(2), icon: Zap, color: 'from-blue-500 to-blue-600' },
    { label: 'Final Loss', value: metrics.finalLoss.toFixed(2), icon: TrendingUp, color: 'from-green-500 to-green-600' },
    { label: 'Improvement', value: `${improvementPercent}%`, icon: Zap, color: 'from-purple-500 to-purple-600' },
    { label: 'Training Time', value: `${trainingTime}s`, icon: Database, color: 'from-orange-500 to-orange-600' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-blue-100">
          <TrendingUp className="w-5 h-5 text-blue-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900">Model Performance</h2>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div
              key={idx}
              className="relative overflow-hidden bg-white border border-gray-200 rounded-lg p-4 transition-all duration-300 hover:shadow-lg hover:border-gray-300"
            >
              <div className={`absolute top-0 right-0 w-20 h-20 bg-gradient-to-br ${stat.color} opacity-10 rounded-bl-full`} />
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">{stat.label}</p>
                  <Icon className="w-4 h-4 text-gray-400" />
                </div>
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Details */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-xs font-semibold text-gray-600 mb-1">Epochs</p>
            <p className="text-xl font-bold text-gray-900">{metrics.epochs}</p>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-xs font-semibold text-gray-600 mb-1">Memory</p>
            <div className="flex items-center gap-1">
              <p className="text-xl font-bold text-gray-900">{metrics.memoryUsage.toFixed(2)}</p>
              <span className="text-xs text-gray-600">GB</span>
            </div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-xs font-semibold text-gray-600 mb-1">Device</p>
            <div className="flex items-center gap-1">
              <Cpu className="w-4 h-4 text-gray-700" />
              <p className="text-sm font-bold text-gray-900">{device.toUpperCase()}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <p className="text-sm font-semibold text-gray-900 mb-4">Loss Progression</p>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }} />
            <Bar dataKey="loss" fill="#3b82f6" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
