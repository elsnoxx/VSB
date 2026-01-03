import { apiGet, apiPost } from "./http";
import type { CreateLocationRequest, LocationRow } from "./types";

export function getLocations(signal?: AbortSignal) {
  return apiGet<LocationRow[]>("/api/locations", signal);
}

export function createLocation(req: CreateLocationRequest) {
  // typicky backend vrací created id (Guid)
  return apiPost<CreateLocationRequest, string>("/api/locations", req);
}
