import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDeviceById } from "../../api/devicesApi";
import type { DeviceRow } from "../../api/types";

export function DeviceDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [device, setDevice] = useState<DeviceRow | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

        const data = await getDeviceById(id, ac.signal);
        setDevice(data);
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

  return (
    <div>
      <button onClick={() => navigate(-1)} style={{ marginBottom: 12 }}>
        ← Back
      </button>

      <h2>Device detail</h2>

      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {!loading && !error && device && (
        <div style={{ border: "1px solid #ddd", padding: 16, maxWidth: 500 }}>
          <p><strong>Serial number:</strong> {device.serialNumber}</p>
          <p><strong>Status:</strong> {device.status}</p>
          <p><strong>Device type:</strong> {device.deviceTypeId}</p>
          <p><strong>Location:</strong> {device.currentLocationId ?? "-"}</p>
          <p><strong>Created:</strong> {new Date(device.createdAtUtc).toLocaleString()}</p>
        </div>
      )}
    </div>
  );
}
