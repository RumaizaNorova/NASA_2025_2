import React from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { TrendingUp, BarChart3, Target, Activity } from 'lucide-react';

const PerformanceChart = ({ modelPerformance, selectedMetric, setSelectedMetric }) => {
  const metrics = [
    { id: 'auc_score', label: 'AUC Score', icon: TrendingUp, color: '#3b82f6' },
    { id: 'accuracy', label: 'Accuracy', icon: Target, color: '#10b981' },
    { id: 'precision', label: 'Precision', icon: BarChart3, color: '#f59e0b' },
    { id: 'recall', label: 'Recall', icon: Activity, color: '#ef4444' },
    { id: 'f1_score', label: 'F1 Score', icon: TrendingUp, color: '#8b5cf6' },
  ];

  if (!modelPerformance || modelPerformance.length === 0) {
    return (
      <div className="chart-container">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-shark-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8 text-ocean-400" />
          </div>
          <p className="text-ocean-300">No performance data available</p>
        </div>
      </div>
    );
  }

  // Prepare data for Plotly
  const modelNames = modelPerformance.map(m => m.model_name);
  const values = modelPerformance.map(m => m[selectedMetric]);

  const plotData = [
    {
      x: modelNames,
      y: values,
      type: 'bar',
      marker: {
        color: values.map((_, index) => {
          const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
          return colors[index % colors.length];
        }),
        line: {
          color: '#ffffff',
          width: 1
        }
      },
      text: values.map(v => v.toFixed(3)),
      textposition: 'outside',
      textfont: {
        color: '#ffffff',
        size: 12
      }
    }
  ];

  const plotLayout = {
    title: {
      text: `Model Performance - ${metrics.find(m => m.id === selectedMetric)?.label}`,
      font: { color: '#ffffff', size: 18 }
    },
    xaxis: {
      title: 'Model',
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    yaxis: {
      title: 'Score',
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280',
      range: [0, 1]
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ffffff' },
    margin: { t: 60, r: 30, b: 60, l: 60 },
    showlegend: false
  };

  const plotConfig = {
    displayModeBar: false,
    responsive: true
  };

  return (
    <div className="space-y-6">
      {/* Metric Selector */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Performance Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {metrics.map((metric) => {
            const Icon = metric.icon;
            const isSelected = selectedMetric === metric.id;
            
            return (
              <button
                key={metric.id}
                onClick={() => setSelectedMetric(metric.id)}
                className={`p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-ocean-600/20 border-ocean-500/50 text-white'
                    : 'bg-shark-700/50 border-shark-600 text-ocean-300 hover:bg-shark-600/50'
                }`}
              >
                <Icon className={`w-5 h-5 mx-auto mb-2 ${isSelected ? 'text-ocean-400' : 'text-ocean-500'}`} />
                <div className="text-xs font-medium">{metric.label}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Performance Chart */}
      <div className="chart-container">
        <Plot
          data={plotData}
          layout={plotLayout}
          config={plotConfig}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      {/* Performance Table */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Detailed Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-shark-700">
                <th className="text-left py-2 text-ocean-400 font-medium">Model</th>
                <th className="text-right py-2 text-ocean-400 font-medium">AUC</th>
                <th className="text-right py-2 text-ocean-400 font-medium">Accuracy</th>
                <th className="text-right py-2 text-ocean-400 font-medium">Precision</th>
                <th className="text-right py-2 text-ocean-400 font-medium">Recall</th>
                <th className="text-right py-2 text-ocean-400 font-medium">F1 Score</th>
              </tr>
            </thead>
            <tbody>
              {modelPerformance.map((model, index) => (
                <motion.tr
                  key={model.model_name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border-b border-shark-800/50 hover:bg-shark-700/30"
                >
                  <td className="py-3 text-white font-medium">{model.model_name}</td>
                  <td className="py-3 text-right text-ocean-300">{model.auc_score.toFixed(3)}</td>
                  <td className="py-3 text-right text-ocean-300">{model.accuracy.toFixed(3)}</td>
                  <td className="py-3 text-right text-ocean-300">{model.precision.toFixed(3)}</td>
                  <td className="py-3 text-right text-ocean-300">{model.recall.toFixed(3)}</td>
                  <td className="py-3 text-right text-ocean-300">{model.f1_score.toFixed(3)}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Best Model Highlight */}
      <div className="glass-card rounded-xl p-6 bg-gradient-to-r from-ocean-600/10 to-purple-600/10 border border-ocean-500/30">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-10 h-10 bg-ocean-600/20 rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-ocean-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold">Best Performing Model</h3>
            <p className="text-ocean-300 text-sm">GradientBoosting Classifier</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">0.972</div>
            <div className="text-xs text-ocean-400">AUC Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">92.9%</div>
            <div className="text-xs text-ocean-400">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">93.4%</div>
            <div className="text-xs text-ocean-400">Precision</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">92.9%</div>
            <div className="text-xs text-ocean-400">Recall</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceChart;

