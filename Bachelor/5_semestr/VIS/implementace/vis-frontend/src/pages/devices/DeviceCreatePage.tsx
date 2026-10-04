import React, { useEffect, useState } from "react";
import { createDevice } from "../../api/devicesApi";
import { getDeviceTypes } from "../../api/deviceTypeApi";
import { getLocations } from "../../api/locationsApi";
import type { DeviceType, LocationRow } from "../../api/types";

export function DeviceCreatePage() {
  const [serialNumber, setSerialNumber] = useState("");
  const [deviceTypeId, setDeviceTypeId] = useState("");
  const [status, setStatus] = useState("New");
  // combobox input (user can pick a location by name or leave empty)
  const [locationInput, setLocationInput] = useState("");

  const [deviceTypes, setDeviceTypes] = useState<DeviceType[]>([]);
  const [loadingTypes, setLoadingTypes] = useState(true);

  const [locations, setLocations] = useState<LocationRow[]>([]);
  const [loadingLocations, setLoadingLocations] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdId, setCreatedId] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    (async () => {
      try {
        setLoadingTypes(true);
        const types = await getDeviceTypes(ac.signal);
        setDeviceTypes(types);
        if (types.length > 0) setDeviceTypeId(types[0].id);
      } catch (e: unknown) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoadingTypes(false);
      }
    })();
    return () => ac.abort();
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    (async () => {
      try {
        setLoadingLocations(true);
        const locs = await getLocations(ac.signal);
        setLocations(locs);
      } catch (e: unknown) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoadingLocations(false);
      }
    })();
    return () => ac.abort();
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setCreatedId(null);

    if (!serialNumber.trim()) {
      setError("Serial number is required.");
      return;
    }

    if (!deviceTypeId.trim()) {
      setError("DeviceTypeId is required (select a device type).");
      return;
    }

    setLoading(true);
    try {
      // determine currentLocationId to send to API:
      // - empty input -> null
      // - input exactly matches a location id -> use it
      // - input matches a location name -> use that location's id
      // - otherwise send the raw input (allows entering a GUID manually)
      const locInput = locationInput.trim();
      let currentLocationId: string | null = null;
      if (locInput === "") {
        currentLocationId = null;
      } else if (locations.find((l) => l.id === locInput)) {
        currentLocationId = locInput;
      } else {
        const byName = locations.find((l) => l.name === locInput);
        currentLocationId = byName ? byName.id : locInput;
      }

      const id = await createDevice({
        serialNumber: serialNumber.trim(),
        deviceTypeId: deviceTypeId.trim(),
        status: status.trim(),
        currentLocationId,
      });

      setCreatedId(String(id).replaceAll('"', ""));
      setSerialNumber("");
      setDeviceTypeId(deviceTypes.length > 0 ? deviceTypes[0].id : "");
      setStatus("New");
      setLocationInput("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Create Device</h2>

      <form onSubmit={onSubmit} style={{ maxWidth: 420 }}>
        <div style={{ marginBottom: 12 }}>
          <label>Serial number *</label>
          <input
            value={serialNumber}
            onChange={(e) => setSerialNumber(e.target.value)}
            style={{ width: "100%", padding: 8 }}
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>Device type *</label>
          {loadingTypes ? (
            <div>Loading types…</div>
          ) : (
            <select
              value={deviceTypeId}
              onChange={(e) => setDeviceTypeId(e.target.value)}
              style={{ width: "100%", padding: 8 }}
            >
              <option value="">-- Select device type --</option>
              {deviceTypes.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          )}
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>Status</label>
          <input
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            style={{ width: "100%", padding: 8 }}
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>LocationId (optional)</label>
          {/* combobox: input with datalist of location names (user can leave empty) */}
          {loadingLocations ? (
            <div>Loading locations…</div>
          ) : (
            <>
              <input
                list="locations-list"
                value={locationInput}
                onChange={(e) => setLocationInput(e.target.value)}
                placeholder="Type location name or paste GUID (optional)"
                style={{ width: "100%", padding: 8 }}
              />
              <datalist id="locations-list">
                <option value=""></option>
                {locations.map((l) => (
                  // show name in suggestions; matching by name will resolve to id on submit
                  <option key={l.id} value={l.name} />
                ))}
              </datalist>
            </>
          )}
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Saving…" : "Create"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      {createdId && <p style={{ color: "green" }}>Created device ID: {createdId}</p>}
    </div>
  );
}