import { useState } from "react";
import { predictWaterQuality } from "../services/api";

const initialForm = {
  ph: "",
  Hardness: "",
  Solids: "",
  Chloramines: "",
  Sulfate: "",
  Conductivity: "",
  Organic_carbon: "",
  Trihalomethanes: "",
  Turbidity: "",
};

const fields = [
  { name: "ph", label: "pH", step: "0.01", min: 0, max: 14 },
  { name: "Hardness", label: "Hardness", step: "0.01", min: 0 },
  { name: "Solids", label: "Solids", step: "0.01", min: 0 },
  { name: "Chloramines", label: "Chloramines", step: "0.01", min: 0 },
  { name: "Sulfate", label: "Sulfate", step: "0.01", min: 0 },
  { name: "Conductivity", label: "Conductivity", step: "0.01", min: 0 },
  { name: "Organic_carbon", label: "Organic Carbon", step: "0.01", min: 0 },
  {
    name: "Trihalomethanes",
    label: "Trihalomethanes",
    step: "0.01",
    min: 0,
  },
  { name: "Turbidity", label: "Turbidity", step: "0.01", min: 0 },
];

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

    const missingField = fields.find((field) => formData[field.name] === "");

    if (missingField) {
      setError(`Please enter ${missingField.label}.`);
      return;
    }

    const numericData = Object.fromEntries(
      Object.entries(formData).map(([key, value]) => [key, Number(value)]),
    );

    if (numericData.ph < 0 || numericData.ph > 14) {
      setError("pH must be between 0 and 14.");
      return;
    }

    const negativeField = Object.entries(numericData).find(
      ([, value]) => value < 0,
    );

    if (negativeField) {
      setError("Water-quality values cannot be negative.");
      return;
    }

    try {
      onLoading(true);

      const result = await predictWaterQuality(numericData);

      onResult(result);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      onLoading(false);
    }
  }

  function handleReset() {
    setFormData(initialForm);
    setError("");
    onResult(null);
  }

  return (
    <form onSubmit={handleSubmit}>
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
              step={field.step}
              min={field.min}
              max={field.max}
              value={formData[field.name]}
              onChange={handleChange}
              placeholder={`Enter ${field.label}`}
              className="w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
            />
          </div>
        ))}
      </div>

      {error && (
        <div className="mt-5 rounded-xl border border-red-900/50 bg-red-950/30 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="mt-7 flex gap-3">
        <button
          type="submit"
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
