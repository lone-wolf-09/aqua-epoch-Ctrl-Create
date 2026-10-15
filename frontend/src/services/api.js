
const API_URL = import.meta.env.VITE_API_URL;

export async function predictWaterQuality(data) {
  const response = await fetch(`${API_URL}/predict/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || `Prediction failed (${response.status})`
    );
  }

  return response.json();
}
