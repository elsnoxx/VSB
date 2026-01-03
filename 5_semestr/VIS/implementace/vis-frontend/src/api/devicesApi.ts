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