import type { DeviceType } from "./types";

const base = "/api/device-types";

export async function createDeviceType(
  data: { name: string; description?: string },
  signal?: AbortSignal
): Promise<DeviceType> {
  const res = await fetch(base, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    signal,
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `Request failed: ${res.status}`);
  }

  return res.json();
}

export async function getDeviceTypes(signal?: AbortSignal): Promise<DeviceType[]> {
  const res = await fetch(base, { signal });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `Request failed: ${res.status}`);
  }
  return res.json();
}