import React from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertTriangle } from 'lucide-react';
import type { RiskAssessment } from '../types';

interface RiskDashboardProps {
  riskAssessments: RiskAssessment[];
  topCriticalNodes: RiskAssessment[];
  averageRisk: number;
}

const getRiskColor = (level: string): string => {
  switch (level) {
    case 'LOW':
      return 'text-green-600';
    case 'MEDIUM':
      return 'text-yellow-600';
    case 'HIGH':
      return 'text-orange-600';
    case 'CRITICAL':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
};

const getRiskBgColor = (level: string): string => {
  switch (level) {
    case 'LOW':
      return 'bg-green-50 border-green-200';
    case 'MEDIUM':
      return 'bg-yellow-50 border-yellow-200';
    case 'HIGH':
      return 'bg-orange-50 border-orange-200';
    case 'CRITICAL':
      return 'bg-red-50 border-red-200';
    default:
      return 'bg-gray-50 border-gray-200';
  }
};

export const RiskDashboard: React.FC<RiskDashboardProps> = ({
  riskAssessments,
  topCriticalNodes,
  averageRisk,
}) => {
  const riskData = riskAssessments.slice(0, 10).map((ra) => ({
    name: ra.nodeId.substring(0, 8),
    criticality: (ra.criticalityScore * 100).toFixed(2),
  }));

  const riskDistribution = {
    LOW: riskAssessments.filter((ra) => ra.riskLevel === 'LOW').length,
    MEDIUM: riskAssessments.filter((ra) => ra.riskLevel === 'MEDIUM').length,
    HIGH: riskAssessments.filter((ra) => ra.riskLevel === 'HIGH').length,
    CRITICAL: riskAssessments.filter((ra) => ra.riskLevel === 'CRITICAL').length,
  };

  const distributionData = [
    { name: 'Low', value: riskDistribution.LOW },
    { name: 'Medium', value: riskDistribution.MEDIUM },
    { name: 'High', value: riskDistribution.HIGH },
    { name: 'Critical', value: riskDistribution.CRITICAL },
  ];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Total Nodes</p>
          <p className="text-3xl font-bold text-blue-600">{riskAssessments.length}</p>
        </div>
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Average Risk</p>
          <p className="text-3xl font-bold text-red-600">{(averageRisk * 100).toFixed(1)}%</p>
        </div>
        <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Critical Nodes</p>
          <p className="text-3xl font-bold text-orange-600">{riskDistribution.CRITICAL}</p>
        </div>
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">High Risk</p>
          <p className="text-3xl font-bold text-yellow-600">{riskDistribution.HIGH}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Criticality Scores */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Top 10 Critical Nodes</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={riskData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="criticality" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={distributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Critical Nodes */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
          <h3 className="font-semibold text-gray-900">Top Critical Nodes</h3>
        </div>

        <div className="space-y-3">
          {topCriticalNodes.slice(0, 5).map((node, index) => (
            <div
              key={node.nodeId}
              className={`p-4 border rounded-lg ${getRiskBgColor(node.riskLevel)}`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center">
                    <span className="text-sm font-bold text-gray-500 mr-3">#{index + 1}</span>
                    <h4 className="font-medium text-gray-900">{node.nodeId}</h4>
                    <span className={`ml-2 px-2 py-1 text-xs font-semibold rounded ${getRiskColor(node.riskLevel)} bg-white`}>
                      {node.riskLevel}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    Criticality: {(node.criticalityScore * 100).toFixed(2)}%
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Price: ${node.pricePrediction.toFixed(2)}</p>
                  <p className="text-sm text-gray-500">Change: {(node.changeForecasted * 100).toFixed(2)}%</p>
                </div>
              </div>
              {node.recommendations.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-300">
                  <p className="text-xs font-semibold text-gray-700 mb-1">Recommendations:</p>
                  <ul className="text-xs text-gray-600 space-y-1">
                    {node.recommendations.map((rec, i) => (
                      <li key={i} className="flex items-start">
                        <span className="mr-2">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
