import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { Target, TrendingUp, Activity, BarChart3 } from 'lucide-react';

const FeatureImportance = () => {
  const [selectedModel, setSelectedModel] = useState('gradientboosting');

  // Mock feature importance data - in a real app, this would come from the API
  const featureImportanceData = {
    gradientboosting: {
      features: [
        'latitude', 'longitude', 'sst', 'chlorophyll_a', 'primary_productivity',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_year_sin',
        'day_of_year_cos', 'distance_to_coast', 'ssh_anomaly', 'sst_anomaly',
        'chl_anomaly', 'pp_anomaly', 'year', 'day_of_week', 'is_weekend'
      ],
      importance: [
        0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01
      ]
    },
    randomforest: {
      features: [
        'latitude', 'longitude', 'sst', 'chlorophyll_a', 'primary_productivity',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_year_sin',
        'day_of_year_cos', 'distance_to_coast', 'ssh_anomaly', 'sst_anomaly',
        'chl_anomaly', 'pp_anomaly', 'year', 'day_of_week', 'is_weekend'
      ],
      importance: [
        0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01
      ]
    },
    logisticregression: {
      features: [
        'latitude', 'longitude', 'sst', 'chlorophyll_a', 'primary_productivity',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_year_sin',
        'day_of_year_cos', 'distance_to_coast', 'ssh_anomaly', 'sst_anomaly',
        'chl_anomaly', 'pp_anomaly', 'year', 'day_of_week', 'is_weekend'
      ],
      importance: [
        0.13, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03,
        0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01
      ]
    }
  };

  const models = [
    { id: 'gradientboosting', label: 'GradientBoosting', icon: TrendingUp },
    { id: 'randomforest', label: 'Random Forest', icon: Target },
    { id: 'logisticregression', label: 'Logistic Regression', icon: BarChart3 }
  ];

  const currentData = featureImportanceData[selectedModel];
  
  // Prepare chart data
  const plotData = [
    {
      x: currentData.importance,
      y: currentData.features,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: currentData.importance,
        colorscale: [
          [0, '#1e40af'],
          [0.25, '#3b82f6'],
          [0.5, '#60a5fa'],
          [0.75, '#93c5fd'],
          [1, '#dbeafe']
        ],
        line: {
          color: '#ffffff',
          width: 1
        }
      },
      text: currentData.importance.map(val => val.toFixed(3)),
      textposition: 'outside',
      textfont: {
        color: '#ffffff',
        size: 10
      }
    }
  ];

  const plotLayout = {
    title: {
      text: `Feature Importance - ${models.find(m => m.id === selectedModel)?.label}`,
      font: { color: '#ffffff', size: 18 }
    },
    xaxis: {
      title: 'Importance Score',
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    yaxis: {
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ffffff' },
    margin: { t: 60, r: 80, b: 60, l: 120 },
    height: 600
  };

  const plotConfig = {
    displayModeBar: false,
    responsive: true
  };

  // Feature categories for better understanding
  const featureCategories = {
    'Geographic': ['latitude', 'longitude', 'distance_to_coast'],
    'Environmental': ['sst', 'chlorophyll_a', 'primary_productivity', 'ssh_anomaly', 'sst_anomaly', 'chl_anomaly', 'pp_anomaly'],
    'Temporal': ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'year', 'day_of_week', 'is_weekend']
  };

  const getFeatureCategory = (feature) => {
    for (const [category, features] of Object.entries(featureCategories)) {
      if (features.includes(feature)) {
        return category;
      }
    }
    return 'Other';
  };

  const getCategoryColor = (category) => {
    const colors = {
      'Geographic': '#3b82f6',
      'Environmental': '#10b981',
      'Temporal': '#f59e0b',
      'Other': '#8b5cf6'
    };
    return colors[category] || '#6b7280';
  };

  return (
    <div className="space-y-6">
      {/* Model Selector */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Feature Importance Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {models.map((model) => {
            const Icon = model.icon;
            const isSelected = selectedModel === model.id;
            
            return (
              <button
                key={model.id}
                onClick={() => setSelectedModel(model.id)}
                className={`p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-ocean-600/20 border-ocean-500/50 text-white'
                    : 'bg-shark-700/50 border-shark-600 text-ocean-300 hover:bg-shark-600/50'
                }`}
              >
                <Icon className={`w-5 h-5 mx-auto mb-2 ${isSelected ? 'text-ocean-400' : 'text-ocean-500'}`} />
                <div className="text-xs font-medium">{model.label}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Feature Importance Chart */}
      <div className="chart-container">
        <Plot
          data={plotData}
          layout={plotLayout}
          config={plotConfig}
          style={{ width: '100%', height: '600px' }}
        />
      </div>

      {/* Feature Categories */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Feature Categories</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(featureCategories).map(([category, features]) => (
            <div key={category} className="space-y-2">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getCategoryColor(category) }}
                ></div>
                <span className="text-white font-medium">{category}</span>
              </div>
              <div className="space-y-1">
                {features.map(feature => {
                  const importance = currentData.importance[currentData.features.indexOf(feature)];
                  return (
                    <div key={feature} className="flex justify-between items-center text-sm">
                      <span className="text-ocean-300">{feature}</span>
                      <span className="text-white font-medium">{importance.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Features */}
      <div className="glass-card rounded-xl p-4 bg-gradient-to-r from-ocean-600/10 to-purple-600/10 border border-ocean-500/30">
        <div className="flex items-start space-x-3">
          <div className="w-8 h-8 bg-ocean-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <Target className="w-4 h-4 text-ocean-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold mb-2">Top 5 Most Important Features</h3>
            <div className="space-y-2">
              {currentData.features.slice(0, 5).map((feature, index) => {
                const importance = currentData.importance[index];
                const category = getFeatureCategory(feature);
                return (
                  <div key={feature} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-ocean-400 font-medium">#{index + 1}</span>
                      <span className="text-white">{feature}</span>
                      <span 
                        className="text-xs px-2 py-1 rounded-full"
                        style={{ 
                          backgroundColor: getCategoryColor(category) + '20',
                          color: getCategoryColor(category)
                        }}
                      >
                        {category}
                      </span>
                    </div>
                    <span className="text-ocean-300 font-medium">{importance.toFixed(3)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Insights */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Feature Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <div className="p-3 bg-blue-600/20 border border-blue-500/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Activity className="w-4 h-4 text-blue-400" />
                <span className="text-blue-400 font-medium">Geographic Features</span>
              </div>
              <p className="text-sm text-blue-300">
                Latitude and longitude are consistently the most important features, indicating strong spatial patterns in shark foraging behavior.
              </p>
            </div>
            
            <div className="p-3 bg-green-600/20 border border-green-500/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-4 h-4 text-green-400" />
                <span className="text-green-400 font-medium">Environmental Features</span>
              </div>
              <p className="text-sm text-green-300">
                Sea surface temperature and chlorophyll-a levels significantly influence shark foraging patterns.
              </p>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="p-3 bg-yellow-600/20 border border-yellow-500/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <BarChart3 className="w-4 h-4 text-yellow-400" />
                <span className="text-yellow-400 font-medium">Temporal Features</span>
              </div>
              <p className="text-sm text-yellow-300">
                Cyclical time features (hour, month, day of year) capture seasonal and diurnal patterns in shark behavior.
              </p>
            </div>
            
            <div className="p-3 bg-purple-600/20 border border-purple-500/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Target className="w-4 h-4 text-purple-400" />
                <span className="text-purple-400 font-medium">Model Performance</span>
              </div>
              <p className="text-sm text-purple-300">
                GradientBoosting shows the best feature importance distribution with clear separation between important and less important features.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureImportance;