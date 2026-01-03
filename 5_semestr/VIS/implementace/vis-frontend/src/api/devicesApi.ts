import { apiGet, apiPost } from "./http";
import type { DeviceRow, CreateDeviceRequest } from "./types";

export function getDevices(signal?: AbortSignal) {
  return apiGet<DeviceRow[]>("/api/devices", signal);
}

export function createDevice(req: CreateDeviceRequest) {
  // backend vrací Guid
  return apiPost<CreateDeviceRequest, string>("/api/devices", req);
}

export function getDeviceById(id: string, signal?: AbortSignal) {
  return apiGet<DeviceRow>(`/api/devices/${id}`, signal);
}

export async function assignDeviceLocation(deviceId: string, locationId: string | null) {
  const res = await fetch(`/api/devices/${deviceId}/assign-location`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ locationId } satisfies AssignDeviceLocationRequest),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
}

async function parseError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  return text || `${res.status} ${res.statusText}`;
}

export async function deleteDevice(id: string): Promise<void> {
  const res = await fetch(`/api/devices/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseError(res));
}

