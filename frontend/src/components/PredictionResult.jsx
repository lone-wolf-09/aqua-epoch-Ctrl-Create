function PredictionResult({ result, loading }) {
  if (loading) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <div className="text-center">
          <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-slate-700 border-t-cyan-400" />

          <p className="font-medium text-slate-300">Analyzing sample...</p>

          <p className="mt-1 text-sm text-slate-500">
            Running the ML prediction
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <div className="text-center">
          <div className="mb-4 text-6xl">💧</div>

          <h4 className="text-lg font-semibold text-slate-300">
            Ready to analyze
          </h4>

          <p className="mt-2 max-w-sm text-sm text-slate-500">
            Enter the water parameters and click{" "}
            <span className="text-slate-300">Analyze Water</span> to receive a
            prediction.
          </p>
        </div>
      </div>
    );
  }

  const confidence = Math.round(result.confidence * 100);
  const isSafe = result.prediction === 1;

  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center">
      {/* Status */}
      <div
        className={`flex h-28 w-28 items-center justify-center rounded-full border-4 ${
          isSafe
            ? "border-emerald-500/40 bg-emerald-500/10"
            : "border-red-500/40 bg-red-500/10"
        }`}
      >
        <span className="text-5xl">{isSafe ? "✓" : "!"}</span>
      </div>

      <h4
        className={`mt-6 text-3xl font-bold ${
          isSafe ? "text-emerald-400" : "text-red-400"
        }`}
      >
        {result.status}
      </h4>

      <p className="mt-2 text-sm text-slate-500">
        ML-based water quality assessment
      </p>

      {/* Confidence */}
      <div className="mt-10 w-full max-w-md">
        <div className="mb-2 flex justify-between">
          <span className="text-sm font-medium text-slate-300">
            Model Score
          </span>

          <span className="text-sm font-semibold text-white">
            {confidence}%
          </span>
        </div>

        <div className="h-3 overflow-hidden rounded-full bg-slate-800">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              isSafe ? "bg-emerald-500" : "bg-red-500"
            }`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      <div className="mt-8 rounded-xl border border-slate-800 bg-slate-950/60 px-5 py-4 text-center">
        <p className="text-sm text-slate-400">
          Prediction:
          <span className="ml-2 font-semibold text-white">
            {result.prediction}
          </span>
        </p>
      </div>
    </div>
  );
}

export default PredictionResult;
