import React, { useEffect, useState, useRef } from 'react';
import { Loader, AlertCircle } from 'lucide-react';

/**
 * Blocks the UI until shark tracks are loaded so we never show a misleading empty map.
 * Explains Render-style cold starts (can take several minutes on first open).
 */
export default function InitialDataOverlay({
  loading,
  error,
  onRetry,
}) {
  const [elapsedSec, setElapsedSec] = useState(0);
  const loadStartedAtRef = useRef(null);

  useEffect(() => {
    if (loading && !error) {
      if (loadStartedAtRef.current == null) {
        loadStartedAtRef.current = Date.now();
      }
    } else {
      loadStartedAtRef.current = null;
    }
  }, [loading, error]);

  useEffect(() => {
    if (!loading || error) return undefined;
    const t = setInterval(() => {
      const base = loadStartedAtRef.current ?? Date.now();
      setElapsedSec(Math.floor((Date.now() - base) / 1000));
    }, 1000);
    return () => clearInterval(t);
  }, [loading, error]);

  if (error) {
    return (
      <div className="absolute inset-0 z-[100] flex items-center justify-center bg-shark-950/95 backdrop-blur-md px-6">
        <div className="max-w-md w-full rounded-2xl border border-shark-600 bg-shark-900/90 p-8 text-center shadow-2xl">
          <AlertCircle className="w-14 h-14 text-amber-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Could not load shark data</h2>
          <p className="text-ocean-200 text-sm mb-6 leading-relaxed">{error}</p>
          <button
            type="button"
            onClick={onRetry}
            className="w-full py-3 rounded-xl bg-ocean-600 hover:bg-ocean-500 text-white font-medium transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-[100] flex items-center justify-center bg-shark-950/90 backdrop-blur-sm px-6">
      <div className="max-w-lg w-full text-center">
        <Loader className="w-14 h-14 text-ocean-400 animate-spin mx-auto mb-6" />
        <h2 className="text-xl font-semibold text-white mb-3">Loading shark tracks</h2>
        <p className="text-ocean-200 text-sm leading-relaxed mb-2">
          The map needs data from the API. If nobody has used the app recently, the server may be
          <span className="text-white font-medium"> starting up (often 1–3 minutes on free hosting)</span>.
          Please keep this page open.
        </p>
        <p className="text-ocean-400 text-xs mb-1">Elapsed: {elapsedSec}s</p>
        <p className="text-shark-500 text-xs">This screen disappears automatically when data is ready.</p>
      </div>
    </div>
  );
}
