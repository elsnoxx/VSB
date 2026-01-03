import { apiGet, apiPost } from "./http";
import type { CreateLocationRequest, LocationRow } from "./types";

export function getLocations(signal?: AbortSignal) {
  return apiGet<LocationRow[]>("/api/locations", signal);
}

export function createLocation(req: CreateLocationRequest) {
  // typicky backend vrací created id (Guid)
  return apiPost<CreateLocationRequest, string>("/api/locations", req);
}

async function parseError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  return text || `${res.status} ${res.statusText}`;
}

export async function deleteLocation(id: string): Promise<void> {
  const res = await fetch(`/api/locations/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await parseError(res));
}
