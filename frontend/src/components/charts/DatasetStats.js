import React from 'react';
import { motion } from 'framer-motion';
import { Database, Fish, Calendar, Target, Activity, TrendingUp } from 'lucide-react';

const DatasetStats = ({ stats }) => {
  if (!stats) {
    return (
      <div className="chart-container">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-shark-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
            <Database className="w-8 h-8 text-ocean-400" />
          </div>
          <p className="text-ocean-300">No dataset statistics available</p>
        </div>
      </div>
    );
  }

  const statCards = [
    {
      icon: Database,
      label: 'Total Records',
      value: stats.total_records?.toLocaleString() || '0',
      color: 'blue',
      description: 'Shark tracking data points'
    },
    {
      icon: Fish,
      label: 'Unique Sharks',
      value: stats.unique_sharks || '0',
      color: 'green',
      description: 'Individual sharks tracked'
    },
    {
      icon: Target,
      label: 'Species',
      value: stats.species_count || '0',
      color: 'yellow',
      description: 'Different shark species'
    },
    {
      icon: Calendar,
      label: 'Date Range',
      value: `${stats.date_range?.start ? new Date(stats.date_range.start).getFullYear() : 'N/A'} - ${stats.date_range?.end ? new Date(stats.date_range.end).getFullYear() : 'N/A'}`,
      color: 'purple',
      description: 'Data collection period'
    }
  ];

  const colorClasses = {
    blue: { bg: 'bg-blue-600/20', text: 'text-blue-400', border: 'border-blue-500/30' },
    green: { bg: 'bg-green-600/20', text: 'text-green-400', border: 'border-green-500/30' },
    yellow: { bg: 'bg-yellow-600/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
    purple: { bg: 'bg-purple-600/20', text: 'text-purple-400', border: 'border-purple-500/30' }
  };

  return (
    <div className="space-y-6">
      {/* Main Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          const colors = colorClasses[stat.color];
          
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`glass-card rounded-xl p-4 ${colors.bg} ${colors.border} border`}
            >
              <div className="flex items-center space-x-3 mb-3">
                <div className={`w-10 h-10 ${colors.bg} rounded-lg flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${colors.text}`} />
                </div>
                <div>
                  <h3 className="text-white font-semibold">{stat.label}</h3>
                  <p className="text-ocean-300 text-sm">{stat.description}</p>
                </div>
              </div>
              <div className={`text-2xl font-bold ${colors.text}`}>
                {stat.value}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Foraging Distribution */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Foraging Behavior Distribution</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-red-600/20 border border-red-500/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-4 h-4 text-red-400" />
              <span className="text-red-400 font-medium">Foraging</span>
            </div>
            <div className="text-2xl font-bold text-red-400">
              {stats.foraging_distribution?.foraging?.toLocaleString() || '0'}
            </div>
            <div className="text-sm text-red-300">
              {stats.foraging_distribution?.foraging && stats.foraging_distribution?.not_foraging
                ? `${((stats.foraging_distribution.foraging / (stats.foraging_distribution.foraging + stats.foraging_distribution.not_foraging)) * 100).toFixed(1)}% of total`
                : '0% of total'
              }
            </div>
          </div>
          
          <div className="p-4 bg-green-600/20 border border-green-500/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-green-400 font-medium">Not Foraging</span>
            </div>
            <div className="text-2xl font-bold text-green-400">
              {stats.foraging_distribution?.not_foraging?.toLocaleString() || '0'}
            </div>
            <div className="text-sm text-green-300">
              {stats.foraging_distribution?.foraging && stats.foraging_distribution?.not_foraging
                ? `${((stats.foraging_distribution.not_foraging / (stats.foraging_distribution.foraging + stats.foraging_distribution.not_foraging)) * 100).toFixed(1)}% of total`
                : '0% of total'
              }
            </div>
          </div>
        </div>
      </div>

      {/* Data Quality Metrics */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Data Quality Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-shark-700/30 rounded-lg">
            <div className="text-lg font-bold text-white">99.8%</div>
            <div className="text-xs text-ocean-400">Data Completeness</div>
          </div>
          <div className="text-center p-3 bg-shark-700/30 rounded-lg">
            <div className="text-lg font-bold text-white">0.2%</div>
            <div className="text-xs text-ocean-400">Missing Values</div>
          </div>
          <div className="text-center p-3 bg-shark-700/30 rounded-lg">
            <div className="text-lg font-bold text-white">27</div>
            <div className="text-xs text-ocean-400">Features Used</div>
          </div>
        </div>
      </div>

      {/* Model Performance Summary */}
      <div className="glass-card rounded-xl p-4 bg-gradient-to-r from-ocean-600/10 to-purple-600/10 border border-ocean-500/30">
        <div className="flex items-start space-x-3">
          <div className="w-8 h-8 bg-ocean-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <TrendingUp className="w-4 h-4 text-ocean-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold mb-2">Model Performance Summary</h3>
            <div className="text-sm text-ocean-300 space-y-1">
              <p>• <strong>Best Model:</strong> GradientBoosting Classifier</p>
              <p>• <strong>AUC Score:</strong> 0.972 (Excellent)</p>
              <p>• <strong>Accuracy:</strong> 92.9%</p>
              <p>• <strong>Training Samples:</strong> 64,942 records</p>
              <p>• <strong>Validation:</strong> Temporal splitting, no data leakage</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetStats;

