import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useShark } from '../context/SharkContext';
import PerformanceChart from './charts/PerformanceChart';
import SpeciesDistribution from './charts/SpeciesDistribution';
import TemporalAnalysis from './charts/TemporalAnalysis';
import FeatureImportance from './charts/FeatureImportance';
import DatasetStats from './charts/DatasetStats';
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  Target, 
  Activity,
  Calendar,
  Fish,
  Zap
} from 'lucide-react';

const Dashboard = () => {
  const { 
    modelPerformance, 
    stats, 
    sharkTracks, 
    species, 
    loading, 
    error 
  } = useShark();

  const [activeTab, setActiveTab] = useState('overview');
  const [selectedMetric, setSelectedMetric] = useState('auc_score');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'performance', label: 'Model Performance', icon: TrendingUp },
    { id: 'species', label: 'Species Analysis', icon: Fish },
    { id: 'temporal', label: 'Temporal Analysis', icon: Calendar },
    { id: 'features', label: 'Feature Importance', icon: Target },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin w-12 h-12 border-4 border-ocean-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-ocean-300">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="w-12 h-12 bg-red-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Activity className="w-6 h-6 text-red-400" />
          </div>
          <p className="text-red-400">Error loading dashboard: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto pt-16 pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">Analytics Dashboard</h1>
          <p className="text-ocean-300">Comprehensive analysis of shark habitat prediction model</p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-ocean-400 text-sm font-medium">Total Records</p>
                <p className="text-2xl font-bold text-white">
                  {stats?.total_records?.toLocaleString() || '0'}
                </p>
              </div>
              <div className="w-12 h-12 bg-ocean-600/20 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-ocean-400" />
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-ocean-400 text-sm font-medium">Unique Sharks</p>
                <p className="text-2xl font-bold text-white">
                  {stats?.unique_sharks || '0'}
                </p>
              </div>
              <div className="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center">
                <Fish className="w-6 h-6 text-green-400" />
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-ocean-400 text-sm font-medium">Species</p>
                <p className="text-2xl font-bold text-white">
                  {stats?.species_count || '0'}
                </p>
              </div>
              <div className="w-12 h-12 bg-yellow-600/20 rounded-lg flex items-center justify-center">
                <Target className="w-6 h-6 text-yellow-400" />
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-ocean-400 text-sm font-medium">Best Model AUC</p>
                <p className="text-2xl font-bold text-white">
                  {modelPerformance?.[0]?.auc_score?.toFixed(3) || '0.000'}
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <div className="flex space-x-1 bg-shark-800/50 p-1 rounded-lg">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 rounded-md transition-all ${
                    isActive
                      ? 'bg-ocean-600 text-white shadow-lg'
                      : 'text-ocean-300 hover:text-white hover:bg-shark-700/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </motion.div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-8"
        >
          {activeTab === 'overview' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <DatasetStats stats={stats} />
              <SpeciesDistribution 
                species={species} 
                speciesDistribution={stats?.species_distribution}
                foragingDistribution={stats?.foraging_distribution}
              />
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="space-y-8">
              <PerformanceChart 
                modelPerformance={modelPerformance}
                selectedMetric={selectedMetric}
                setSelectedMetric={setSelectedMetric}
              />
            </div>
          )}

          {activeTab === 'species' && (
            <div className="space-y-8">
              <SpeciesDistribution 
                species={species} 
                speciesDistribution={stats?.species_distribution}
                foragingDistribution={stats?.foraging_distribution}
                detailed={true}
              />
            </div>
          )}

          {activeTab === 'temporal' && (
            <div className="space-y-8">
              <TemporalAnalysis sharkTracks={sharkTracks} />
            </div>
          )}

          {activeTab === 'features' && (
            <div className="space-y-8">
              <FeatureImportance />
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;

