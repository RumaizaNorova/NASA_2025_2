import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { Fish, Target, Activity, PieChart } from 'lucide-react';

const SpeciesDistribution = ({ 
  species, 
  speciesDistribution, 
  foragingDistribution, 
  detailed = false 
}) => {
  const [selectedSpecies, setSelectedSpecies] = useState(null);

  if (!speciesDistribution && !foragingDistribution) {
    return (
      <div className="chart-container">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-shark-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
            <Fish className="w-8 h-8 text-ocean-400" />
          </div>
          <p className="text-ocean-300">No species distribution data available</p>
        </div>
      </div>
    );
  }

  // Prepare pie chart data for species distribution
  const speciesData = Object.entries(speciesDistribution || {}).map(([name, count]) => ({
    name: name.replace(' (Galeocerdo cuvier)', '').replace(' (Carcharodon carcharias)', ''),
    count
  }));

  const pieData = [
    {
      values: speciesData.map(s => s.count),
      labels: speciesData.map(s => s.name),
      type: 'pie',
      hole: 0.4,
      marker: {
        colors: [
          '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
          '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'
        ],
        line: {
          color: '#ffffff',
          width: 2
        }
      },
      textinfo: 'label+percent',
      textfont: {
        color: '#ffffff',
        size: 12
      }
    }
  ];

  const pieLayout = {
    title: {
      text: 'Species Distribution',
      font: { color: '#ffffff', size: 16 }
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ffffff' },
    margin: { t: 40, r: 20, b: 40, l: 20 },
    showlegend: true,
    legend: {
      x: 1,
      y: 1,
      font: { color: '#ffffff', size: 10 }
    }
  };

  // Prepare bar chart data for foraging distribution
  const foragingData = [
    {
      x: ['Not Foraging', 'Foraging'],
      y: [foragingDistribution?.not_foraging || 0, foragingDistribution?.foraging || 0],
      type: 'bar',
      marker: {
        color: ['#10b981', '#ef4444'],
        line: {
          color: '#ffffff',
          width: 1
        }
      },
      text: [foragingDistribution?.not_foraging || 0, foragingDistribution?.foraging || 0],
      textposition: 'outside',
      textfont: {
        color: '#ffffff',
        size: 12
      }
    }
  ];

  const barLayout = {
    title: {
      text: 'Foraging Behavior Distribution',
      font: { color: '#ffffff', size: 16 }
    },
    xaxis: {
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
    margin: { t: 40, r: 30, b: 40, l: 60 },
    showlegend: false
  };

  const plotConfig = {
    displayModeBar: false,
    responsive: true
  };

  return (
    <div className="space-y-6">
      {/* Species Pie Chart */}
      <div className="chart-container">
        <Plot
          data={pieData}
          layout={pieLayout}
          config={plotConfig}
          style={{ width: '100%', height: '400px' }}
        />
      </div>

      {/* Foraging Distribution */}
      <div className="chart-container">
        <Plot
          data={foragingData}
          layout={barLayout}
          config={plotConfig}
          style={{ width: '100%', height: '300px' }}
        />
      </div>

      {/* Species List */}
      <div className="glass-card rounded-xl p-4">
        <h3 className="text-white font-semibold mb-4">Species Overview</h3>
        <div className="space-y-3">
          {speciesData.map((species, index) => (
            <motion.div
              key={species.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                selectedSpecies === species.name
                  ? 'bg-ocean-600/20 border-ocean-500/50'
                  : 'bg-shark-700/30 border-shark-600 hover:bg-shark-600/30'
              }`}
              onClick={() => setSelectedSpecies(
                selectedSpecies === species.name ? null : species.name
              )}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 rounded-full bg-ocean-400"></div>
                  <div>
                    <div className="text-white font-medium">{species.name}</div>
                    <div className="text-ocean-400 text-sm">
                      {((species.count / Object.values(speciesDistribution).reduce((a, b) => a + b, 0)) * 100).toFixed(1)}% of total
                    </div>
                  </div>
                </div>
                <div className="text-ocean-300 font-semibold">
                  {species.count.toLocaleString()}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Detailed Species Analysis */}
      {detailed && (
        <div className="glass-card rounded-xl p-4">
          <h3 className="text-white font-semibold mb-4">Species Characteristics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-ocean-400" />
                <span className="text-ocean-300 text-sm">Most Common</span>
              </div>
              <div className="text-white font-semibold">
                {speciesData[0]?.name || 'N/A'}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Activity className="w-4 h-4 text-ocean-400" />
                <span className="text-ocean-300 text-sm">Total Species</span>
              </div>
              <div className="text-white font-semibold">
                {speciesData.length}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SpeciesDistribution;

