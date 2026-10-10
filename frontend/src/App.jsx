import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import PredictionResult from "./components/PredictionResult";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="border-b border-slate-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              💧 AquaSense
            </h1>

            <p className="mt-1 text-sm text-slate-400">
              AI-powered water quality assessment
            </p>
          </div>

          <div className="hidden rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300 sm:block">
            ML Prediction System
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-10">
          <h2 className="text-3xl font-bold">
            Analyze Water Quality
          </h2>

          <p className="mt-2 max-w-2xl text-slate-400">
            Enter the measured water-quality parameters to estimate
            whether the sample is potable.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            <div className="mb-6">
              <h3 className="text-xl font-semibold">
                Water Parameters
              </h3>

              <p className="mt-1 text-sm text-slate-400">
                Provide values for all measured parameters.
              </p>
            </div>

            <PredictionForm
              onResult={setResult}
              onLoading={setLoading}
            />
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
            <div className="mb-6">
              <h3 className="text-xl font-semibold">
                Analysis Result
              </h3>

              <p className="mt-1 text-sm text-slate-400">
                Your model prediction will appear here.
              </p>
            </div>

            <PredictionResult
              result={result}
              loading={loading}
            />
          </section>
        </div>

        <div className="mt-8 rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-sm text-slate-500">
          <strong className="text-slate-400">Note:</strong>{" "}
          This system provides an ML-based assessment and is not a
          substitute for laboratory testing or official water-quality
          certification.
        </div>
      </main>
    </div>
  );
}

export default App;