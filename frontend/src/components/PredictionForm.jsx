import { useState } from "react";
import { predictWaterQuality } from "../services/api.js";

const fields = [
  { name: "pH", label: "pH", min: 0, max: 14 },
  { name: "Hardness", label: "Hardness", min: 0 },
  { name: "TDS", label: "Total Dissolved Solids (TDS)", min: 0 },
  { name: "Chloramines", label: "Chloramines", min: 0 },
  { name: "Sulfate", label: "Sulfate", min: 0 },
  { name: "Conductivity", label: "Conductivity", min: 0 },
  { name: "Organic_carbon", label: "Organic Carbon", min: 0 },
  { name: "Trihalomethanes", label: "Trihalomethanes", min: 0 },
  { name: "Turbidity", label: "Turbidity", min: 0 },
  { name: "Calcium", label: "Calcium", min: 0 },
  { name: "Magnesium", label: "Magnesium", min: 0 },
  { name: "Sodium", label: "Sodium", min: 0 },
  { name: "Potassium", label: "Potassium", min: 0 },
  { name: "Chloride", label: "Chloride", min: 0 },
  { name: "Nitrate", label: "Nitrate", min: 0 },
  { name: "Fluoride", label: "Fluoride", min: 0 },
  { name: "Iron", label: "Iron", min: 0 },
  { name: "Alkalinity", label: "Alkalinity", min: 0 },
];

const sampleA = {
  pH: "7.370451475",
  Hardness: "194.6026196",
  TDS: "27809.24393",
  Chloramines: "6.88476349",
  Sulfate: "330.3923698",
  Conductivity: "556.2005271",
  Organic_carbon: "22.63033458",
  Trihalomethanes: "82.04832728",
  Turbidity: "4.969195281",
  Calcium: "71.53226144",
  Magnesium: "31.44947295",
  Sodium: "94.06644811",
  Potassium: "2.925427173",
  Chloride: "210.3000667",
  Nitrate: "58.02405171",
  Fluoride: "1.501053471",
  Iron: "0.2795643829",
  Alkalinity: "127.9791848",
};

const sampleB = {
  pH: "6.986068836",
  Hardness: "189.3700643",
  TDS: "18749.86476",
  Chloramines: "6.953856279",
  Sulfate: "395.1901089",
  Conductivity: "367.3831467",
  Organic_carbon: "15.28755755",
  Trihalomethanes: "86.69757442",
  Turbidity: "4.826220898",
  Calcium: "67.37551174",
  Magnesium: "30.6359168",
  Sodium: "52.76781101",
  Potassium: "7.096351043",
  Chloride: "134.5999706",
  Nitrate: "9.56870303",
  Fluoride: "0.625342565",
  Iron: "0.4202811074",
  Alkalinity: "138.1821916",
};

const initialForm = Object.fromEntries(fields.map(({ name }) => [name, ""]));

function PredictionForm({ onResult, onLoading }) {
  const [formData, setFormData] = useState(initialForm);
  const [error, setError] = useState("");

  function handleChange(event) {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const missing = fields.find(({ name }) => formData[name].trim() === "");

    if (missing) {
      setError(`Please enter ${missing.label}.`);
      return;
    }

    const numericData = Object.fromEntries(
      fields.map(({ name }) => [name, Number(formData[name])]),
    );

    const invalid = fields.find(({ name, min, max }) => {
      const value = numericData[name];

      return (
        !Number.isFinite(value) ||
        value < min ||
        (max !== undefined && value > max)
      );
    });

    if (invalid) {
      setError(`Please enter a valid value for ${invalid.label}.`);
      return;
    }

    try {
      onLoading(true);
      onResult(null);

      const result = await predictWaterQuality(numericData);
      onResult(result);
    } catch (err) {
      setError(err.message || "Prediction failed. Check the backend.");
    } finally {
      onLoading(false);
    }
  }

  function handleReset() {
    setFormData(initialForm);
    setError("");
    onResult(null);
  }

  function loadSample(sample) {
    setFormData({ ...sample });
    setError("");
    onResult(null);
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="mb-6">
        <p className="mb-3 text-sm font-medium text-slate-300">
          Quick-fill demo samples
        </p>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => loadSample(sampleA)}
            className="rounded-xl border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-sm font-medium text-cyan-300 transition hover:bg-cyan-500/20"
          >
            Sample A
          </button>

          <button
            type="button"
            onClick={() => loadSample(sampleB)}
            className="rounded-xl border border-violet-500/40 bg-violet-500/10 px-4 py-2 text-sm font-medium text-violet-300 transition hover:bg-violet-500/20"
          >
            Sample B
          </button>
        </div>

        <p className="mt-2 text-xs text-slate-500">
          Select a sample to populate all 18 parameters, then click Analyze
          Water. These are demo inputs, not certified safe or unsafe samples.
        </p>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        {fields.map((field) => (
          <div key={field.name}>
            <label
              htmlFor={field.name}
              className="mb-2 block text-sm font-medium text-slate-300"
            >
              {field.label}
            </label>

            <input
              id={field.name}
              name={field.name}
              type="number"
              step="any"
              min={field.min}
              max={field.max}
              required
              value={formData[field.name]}
              onChange={handleChange}
              placeholder={`Enter ${field.label}`}
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
            />
          </div>
        ))}
      </div>

      {error && (
        <div
          role="alert"
          className="mt-5 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-300"
        >
          {error}
        </div>
      )}

      <div className="mt-7 flex gap-3">
        <button
          type="submit"
          disabled={false}
          className="flex-1 rounded-xl bg-cyan-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 active:scale-[0.98]"
        >
          Analyze Water
        </button>

        <button
          type="button"
          onClick={handleReset}
          className="rounded-xl border border-slate-700 px-5 py-3 font-medium text-slate-300 transition hover:bg-slate-800"
        >
          Reset
        </button>
      </div>
    </form>
  );
}

export default PredictionForm;
