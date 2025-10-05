import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const GlowEffect = ({ 
  children, 
  intensity = 1, 
  color = '#3b82f6', 
  duration = 2,
  size = 'medium' 
}) => {
  const glowRef = useRef(null);

  const sizeClasses = {
    small: 'w-2 h-2',
    medium: 'w-4 h-4',
    large: 'w-6 h-6',
    xlarge: 'w-8 h-8'
  };

  useEffect(() => {
    if (glowRef.current) {
      const element = glowRef.current;
      element.style.setProperty('--glow-color', color);
      element.style.setProperty('--glow-intensity', intensity);
    }
  }, [color, intensity]);

  return (
    <motion.div
      ref={glowRef}
      className="relative inline-block"
      animate={{
        boxShadow: [
          `0 0 ${5 * intensity}px ${color}40`,
          `0 0 ${20 * intensity}px ${color}60`,
          `0 0 ${5 * intensity}px ${color}40`
        ]
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "easeInOut"
      }}
      style={{
        '--glow-color': color,
        '--glow-intensity': intensity
      }}
    >
      {children}
      
      {/* Glow overlay */}
      <div 
        className="absolute inset-0 rounded-full pointer-events-none"
        style={{
          background: `radial-gradient(circle, ${color}20 0%, transparent 70%)`,
          filter: `blur(${2 * intensity}px)`,
          transform: 'scale(1.5)'
        }}
      />
    </motion.div>
  );
};

export const PulsingDot = ({ 
  color = '#3b82f6', 
  size = 'medium', 
  intensity = 1,
  duration = 2 
}) => {
  const sizeClasses = {
    small: 'w-2 h-2',
    medium: 'w-4 h-4',
    large: 'w-6 h-6',
    xlarge: 'w-8 h-8'
  };

  return (
    <motion.div
      className={`${sizeClasses[size]} rounded-full bg-current`}
      style={{ color }}
      animate={{
        scale: [1, 1.2, 1],
        opacity: [0.7, 1, 0.7]
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "easeInOut"
      }}
    />
  );
};

export const RippleEffect = ({ 
  color = '#3b82f6', 
  size = 'medium',
  duration = 1.5 
}) => {
  const sizeClasses = {
    small: 'w-4 h-4',
    medium: 'w-8 h-8',
    large: 'w-12 h-12',
    xlarge: 'w-16 h-16'
  };

  return (
    <div className={`${sizeClasses[size]} relative flex items-center justify-center`}>
      {[0, 1, 2].map((index) => (
        <motion.div
          key={index}
          className="absolute inset-0 rounded-full border-2"
          style={{ borderColor: color }}
          animate={{
            scale: [0, 2],
            opacity: [1, 0]
          }}
          transition={{
            duration: duration,
            repeat: Infinity,
            ease: "easeOut",
            delay: index * 0.2
          }}
        />
      ))}
      <div 
        className={`${sizeClasses[size]} rounded-full bg-current`}
        style={{ color }}
      />
    </div>
  );
};

export const WaveEffect = ({ 
  color = '#3b82f6', 
  width = '100%',
  height = '4px' 
}) => {
  return (
    <div className="relative overflow-hidden" style={{ width, height }}>
      <motion.div
        className="absolute inset-0 bg-current"
        style={{ color }}
        animate={{
          x: ['-100%', '100%']
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "linear"
        }}
      />
    </div>
  );
};

export const FloatingAnimation = ({ 
  children, 
  intensity = 10,
  duration = 3 
}) => {
  return (
    <motion.div
      animate={{
        y: [-intensity, intensity, -intensity]
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "easeInOut"
      }}
    >
      {children}
    </motion.div>
  );
};

export const ShimmerEffect = ({ 
  children, 
  color = '#ffffff',
  duration = 2 
}) => {
  return (
    <motion.div
      className="relative overflow-hidden"
      animate={{
        backgroundPosition: ['200% 0', '-200% 0']
      }}
      transition={{
        duration: duration,
        repeat: Infinity,
        ease: "linear"
      }}
      style={{
        background: `linear-gradient(90deg, transparent, ${color}20, transparent)`,
        backgroundSize: '200% 100%'
      }}
    >
      {children}
    </motion.div>
  );
};

export default GlowEffect;

