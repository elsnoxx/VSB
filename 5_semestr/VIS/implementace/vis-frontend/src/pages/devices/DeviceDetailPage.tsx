import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDeviceById, assignDeviceLocation } from "../../api/devicesApi";
import { getLocations } from "../../api/locationsApi";
import type { DeviceRow, LocationRow } from "../../api/types";

export function DeviceDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [device, setDevice] = useState<DeviceRow | null>(null);
  const [locations, setLocations] = useState<LocationRow[]>([]);
  const [selectedLocationId, setSelectedLocationId] = useState<string>("");

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      setError("Missing device id.");
      return;
    }

    const ac = new AbortController();

    (async () => {
      try {
        setLoading(true);
        setError(null);

        const [deviceData, locationsData] = await Promise.all([
          getDeviceById(id, ac.signal),
          getLocations(ac.signal),
        ]);

        setDevice(deviceData);
        setLocations(locationsData);
        setSelectedLocationId(deviceData.currentLocationId ?? "");
      } catch (e: unknown) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        if (e instanceof Error && e.message.includes("aborted")) return;

        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    })();

    return () => ac.abort();
  }, [id]);

  async function onAssignLocation() {
    if (!id || !device) return;

    setSaving(true);
    setError(null);
    setInfo(null);

    try {
      const locationId = selectedLocationId || null;

      await assignDeviceLocation(id, locationId);

      setDevice({
        ...device,
        currentLocationId: locationId,
      });

      setInfo(
        locationId
          ? "Location successfully assigned."
          : "Location removed from device."
      );
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <button onClick={() => navigate(-1)} style={{ marginBottom: 12 }}>
        ← Back
      </button>

      <h2>Device detail</h2>

      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      {info && <p style={{ color: "green" }}>{info}</p>}

      {!loading && !error && device && (
        <>
          {/* Device info */}
          <div style={{ border: "1px solid #ddd", padding: 16, maxWidth: 600 }}>
            <p><strong>Serial number:</strong> {device.serialNumber}</p>
            <p><strong>Status:</strong> {device.status}</p>
            <p><strong>Device type:</strong> {device.deviceType?.name ?? device.deviceTypeId ?? "-"}</p>
            <p><strong>Current location:</strong> {device.currentLocationId
              ? locations.find((l) => l.id === device.currentLocationId)?.name ?? device.currentLocationId
              : "-"}</p>
            <p><strong>Created:</strong> {new Date(device.createdAtUtc).toLocaleString()}</p>
          </div>

          {/* Assign location */}
          <div
            style={{
              marginTop: 16,
              border: "1px solid #ddd",
              padding: 16,
              maxWidth: 600,
            }}
          >
            <h3 style={{ marginTop: 0 }}>Assign location</h3>

            <div style={{ display: "flex", gap: 12 }}>
              <select
                value={selectedLocationId}
                onChange={(e) => setSelectedLocationId(e.target.value)}
                style={{ padding: 8, flex: 1 }}
              >
                <option value="">— No location —</option>
                {locations.map((l) => (
                  <option key={l.id} value={l.id}>
                    {l.name}
                  </option>
                ))}
              </select>

              <button
                onClick={onAssignLocation}
                disabled={saving}
                style={{ padding: "8px 12px" }}
              >
                {saving ? "Saving…" : "Assign"}
              </button>
            </div>

            <small style={{ display: "block", marginTop: 8, color: "#555" }}>
              Business rule: one location can have at most one assigned device.
            </small>
          </div>
        </>
      )}
    </div>
  );
}
