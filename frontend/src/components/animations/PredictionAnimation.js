import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Target, TrendingUp, Activity } from 'lucide-react';

const PredictionAnimation = ({ 
  predictionData, 
  isAnimating, 
  onAnimationComplete 
}) => {
  const [animationStep, setAnimationStep] = useState(0);
  const [showDetails, setShowDetails] = useState(false);

  const animationSteps = [
    { id: 'loading', label: 'Analyzing Location', icon: Target, color: 'blue' },
    { id: 'processing', label: 'Processing Environmental Data', icon: Activity, color: 'yellow' },
    { id: 'calculating', label: 'Calculating Probability', icon: TrendingUp, color: 'green' },
    { id: 'complete', label: 'Prediction Complete', icon: Zap, color: 'purple' }
  ];

  useEffect(() => {
    if (isAnimating) {
      setAnimationStep(0);
      setShowDetails(false);
      
      const interval = setInterval(() => {
        setAnimationStep(prev => {
          const next = prev + 1;
          if (next >= animationSteps.length) {
            clearInterval(interval);
            setShowDetails(true);
            onAnimationComplete();
            return prev;
          }
          return next;
        });
      }, 800);

      return () => clearInterval(interval);
    }
  }, [isAnimating, onAnimationComplete]);

  const getColorClasses = (color) => {
    const colors = {
      blue: { bg: 'bg-blue-600/20', text: 'text-blue-400', border: 'border-blue-500/30' },
      yellow: { bg: 'bg-yellow-600/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
      green: { bg: 'bg-green-600/20', text: 'text-green-400', border: 'border-green-500/30' },
      purple: { bg: 'bg-purple-600/20', text: 'text-purple-400', border: 'border-purple-500/30' }
    };
    return colors[color] || colors.blue;
  };

  if (!isAnimating && !showDetails) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="fixed inset-0 bg-shark-900/80 backdrop-blur-sm flex items-center justify-center z-50"
      >
        <div className="glass-card rounded-xl p-6 max-w-md w-full mx-4">
          {isAnimating ? (
            <div className="text-center">
              <h3 className="text-white font-semibold mb-6">Generating Prediction</h3>
              
              {/* Animation Steps */}
              <div className="space-y-4 mb-6">
                {animationSteps.map((step, index) => {
                  const Icon = step.icon;
                  const colors = getColorClasses(step.color);
                  const isActive = index === animationStep;
                  const isCompleted = index < animationStep;
                  
                  return (
                    <motion.div
                      key={step.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ 
                        opacity: isActive || isCompleted ? 1 : 0.5,
                        x: 0
                      }}
                      className={`flex items-center space-x-3 p-3 rounded-lg border transition-all ${
                        isActive 
                          ? `${colors.bg} ${colors.border} ${colors.text}`
                          : isCompleted
                          ? 'bg-green-600/20 border-green-500/30 text-green-400'
                          : 'bg-shark-700/30 border-shark-600 text-shark-400'
                      }`}
                    >
                      <div className="relative">
                        {isCompleted ? (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center"
                          >
                            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                          </motion.div>
                        ) : (
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                            isActive ? 'bg-current' : 'bg-shark-600'
                          }`}>
                            {isActive ? (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                              >
                                <Icon className="w-4 h-4 text-white" />
                              </motion.div>
                            ) : (
                              <Icon className="w-4 h-4 text-shark-400" />
                            )}
                          </div>
                        )}
                      </div>
                      
                      <div className="flex-1">
                        <div className="font-medium">{step.label}</div>
                        {isActive && (
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 0.8 }}
                            className="h-1 bg-current rounded-full mt-1"
                          />
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>

              {/* Progress Indicator */}
              <div className="w-full bg-shark-700 rounded-full h-2 mb-4">
                <motion.div
                  className="h-2 bg-gradient-to-r from-ocean-500 to-purple-500 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${((animationStep + 1) / animationSteps.length) * 100}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
              
              <p className="text-ocean-300 text-sm">
                Step {animationStep + 1} of {animationSteps.length}
              </p>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center"
            >
              <div className="w-16 h-16 bg-green-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-green-400" />
              </div>
              
              <h3 className="text-white font-semibold mb-2">Prediction Complete!</h3>
              <p className="text-ocean-300 text-sm mb-4">
                Habitat suitability analysis finished
              </p>
              
              {predictionData && (
                <div className="bg-shark-700/30 rounded-lg p-4 mb-4">
                  <div className="text-2xl font-bold text-white mb-1">
                    {(predictionData.foraging_probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-ocean-300">
                    Foraging Probability
                  </div>
                </div>
              )}
              
              <button
                onClick={() => setShowDetails(false)}
                className="w-full p-3 bg-ocean-600 hover:bg-ocean-700 border border-ocean-500 rounded-lg text-white transition-colors"
              >
                View Results
              </button>
            </motion.div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default PredictionAnimation;

