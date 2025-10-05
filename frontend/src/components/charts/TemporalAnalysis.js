import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { Calendar, Clock, TrendingUp, Activity } from 'lucide-react';

const TemporalAnalysis = ({ sharkTracks }) => {
  const [selectedPeriod, setSelectedPeriod] = useState('month');

  if (!sharkTracks || sharkTracks.length === 0) {
    return (
      <div className="chart-container">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-shark-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
            <Calendar className="w-8 h-8 text-ocean-400" />
          </div>
          <p className="text-ocean-300">No temporal data available</p>
        </div>
      </div>
    );
  }

  // Process temporal data
  const processTemporalData = (tracks, period) => {
    const data = {};
    
    tracks.forEach(track => {
      const date = new Date(track.datetime);
      let key;
      
      switch (period) {
        case 'hour':
          key = date.getHours();
          break;
        case 'day':
          key = date.getDay(); // 0 = Sunday, 1 = Monday, etc.
          break;
        case 'month':
          key = date.getMonth();
          break;
        case 'year':
          key = date.getFullYear();
          break;
        default:
          key = date.getMonth();
      }
      
      if (!data[key]) {
        data[key] = { total: 0, foraging: 0 };
      }
      
      data[key].total++;
      if (track.foraging_behavior === 1) {
        data[key].foraging++;
      }
    });
    
    return data;
  };

  const temporalData = processTemporalData(sharkTracks, selectedPeriod);

  // Prepare chart data
  const getLabels = (period) => {
    switch (period) {
      case 'hour':
        return Array.from({ length: 24 }, (_, i) => `${i}:00`);
      case 'day':
        return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      case 'month':
        return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      case 'year':
        return Object.keys(temporalData).sort();
      default:
        return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    }
  };

  const labels = getLabels(selectedPeriod);
  const totalData = labels.map(label => {
    const key = labels.indexOf(label);
    return temporalData[key]?.total || 0;
  });
  
  const foragingData = labels.map(label => {
    const key = labels.indexOf(label);
    return temporalData[key]?.foraging || 0;
  });

  const foragingRateData = labels.map(label => {
    const key = labels.indexOf(label);
    const data = temporalData[key];
    return data ? (data.foraging / data.total) * 100 : 0;
  });

  const plotData = [
    {
      x: labels,
      y: totalData,
      type: 'bar',
      name: 'Total Tracks',
      marker: {
        color: '#3b82f6',
        line: {
          color: '#ffffff',
          width: 1
        }
      }
    },
    {
      x: labels,
      y: foragingData,
      type: 'bar',
      name: 'Foraging Tracks',
      marker: {
        color: '#ef4444',
        line: {
          color: '#ffffff',
          width: 1
        }
      }
    }
  ];

  const plotLayout = {
    title: {
      text: `Shark Activity by ${selectedPeriod.charAt(0).toUpperCase() + selectedPeriod.slice(1)}`,
      font: { color: '#ffffff', size: 18 }
    },
    xaxis: {
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    yaxis: {
      title: 'Number of Tracks',
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ffffff' },
    margin: { t: 60, r: 30, b: 60, l: 60 },
    barmode: 'group',
    legend: {
      x: 1,
      y: 1,
      font: { color: '#ffffff' }
    }
  };

  const ratePlotData = [
    {
      x: labels,
      y: foragingRateData,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Foraging Rate',
      line: {
        color: '#f59e0b',
        width: 3
      },
      marker: {
        color: '#f59e0b',
        size: 8,
        line: {
          color: '#ffffff',
          width: 2
        }
      }
    }
  ];

  const ratePlotLayout = {
    title: {
      text: `Foraging Rate by ${selectedPeriod.charAt(0).toUpperCase() + selectedPeriod.slice(1)}`,
      font: { color: '#ffffff', size: 18 }
    },
    xaxis: {
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280'
    },
    yaxis: {
      title: 'Foraging Rate (%)',
      color: '#ffffff',
      gridcolor: '#374151',
      linecolor: '#6b7280',
      range: [0, 100]
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

  const periods = [
    { id: 'hour', label: 'Hour of Day', icon: Clock },
    { id: 'day', label: 'Day of Week', icon: Calendar },
    { id: 'month', label: 'Month', icon: TrendingUp },
    { id: 'year', label: 'Year', icon: Activity }
  ];

  return (
    <div className="space-y-6">
      {/* Period Selector */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Temporal Analysis</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {periods.map((period) => {
            const Icon = period.icon;
            const isSelected = selectedPeriod === period.id;
            
            return (
              <button
                key={period.id}
                onClick={() => setSelectedPeriod(period.id)}
                className={`p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-ocean-600/20 border-ocean-500/50 text-white'
                    : 'bg-shark-700/50 border-shark-600 text-ocean-300 hover:bg-shark-600/50'
                }`}
              >
                <Icon className={`w-5 h-5 mx-auto mb-2 ${isSelected ? 'text-ocean-400' : 'text-ocean-500'}`} />
                <div className="text-xs font-medium">{period.label}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Activity Chart */}
      <div className="chart-container">
        <Plot
          data={plotData}
          layout={plotLayout}
          config={plotConfig}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      {/* Foraging Rate Chart */}
      <div className="chart-container">
        <Plot
          data={ratePlotData}
          layout={ratePlotLayout}
          config={plotConfig}
          style={{ width: '100%', height: '300px' }}
        />
      </div>

      {/* Temporal Insights */}
      <div className="glass-card rounded-xl p-4 bg-gradient-to-r from-ocean-600/10 to-purple-600/10 border border-ocean-500/30">
        <div className="flex items-start space-x-3">
          <div className="w-8 h-8 bg-ocean-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
            <TrendingUp className="w-4 h-4 text-ocean-400" />
          </div>
          <div>
            <h3 className="text-white font-semibold mb-2">Temporal Patterns</h3>
            <div className="text-sm text-ocean-300 space-y-1">
              <p>• <strong>Peak Activity:</strong> {(() => {
                const maxIndex = totalData.indexOf(Math.max(...totalData));
                return labels[maxIndex] || 'N/A';
              })()}</p>
              <p>• <strong>Highest Foraging Rate:</strong> {(() => {
                const maxRateIndex = foragingRateData.indexOf(Math.max(...foragingRateData));
                return `${labels[maxRateIndex] || 'N/A'} (${Math.max(...foragingRateData).toFixed(1)}%)`;
              })()}</p>
              <p>• <strong>Total Tracks:</strong> {totalData.reduce((a, b) => a + b, 0).toLocaleString()}</p>
              <p>• <strong>Average Foraging Rate:</strong> {(foragingRateData.reduce((a, b) => a + b, 0) / foragingRateData.length).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TemporalAnalysis;