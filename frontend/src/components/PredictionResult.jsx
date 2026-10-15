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
  const predictedPH = Number(result.predicted_pH);

  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center">
      <div className="flex h-28 w-28 items-center justify-center rounded-full border-4 border-cyan-500/40 bg-cyan-500/10">
        <span className="text-5xl text-cyan-400">◎</span>
      </div>

      <h4 className="mt-6 text-3xl font-bold text-cyan-400">
        {result.prediction === 1 ? "Safe" : "Unsafe"}
      </h4>

      <p className="mt-2 text-sm text-slate-500">
        ML classification result — not a confirmed safety verdict
      </p>

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
            className="h-full rounded-full bg-cyan-500 transition-all duration-700"
            style={{ width: `${confidence}%` }}
          />
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Confidence is the model's estimated probability for its predicted
          class, not a guarantee of water safety.
        </p>
      </div>

      <div className="mt-6 grid w-full max-w-md grid-cols-2 gap-3">
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-center">
          <p className="text-sm text-slate-400">Predicted class</p>
          <p className="mt-1 text-xl font-semibold text-white">
            {result.prediction}
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-center">
          <p className="text-sm text-slate-400">Estimated pH</p>
          <p className="mt-1 text-xl font-semibold text-white">
            {Number.isFinite(predictedPH) ? predictedPH.toFixed(2) : "—"}
          </p>
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;
