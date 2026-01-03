import React, { useState } from "react";
import { createDevice } from "../../api/devicesApi";

export function DeviceCreatePage() {
  const [serialNumber, setSerialNumber] = useState("");
  const [deviceTypeId, setDeviceTypeId] = useState("3fa85f64-5717-4562-b3fc-2c963f66afa6");
  const [status, setStatus] = useState("New");
  const [locationId, setLocationId] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdId, setCreatedId] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setCreatedId(null);

    if (!serialNumber.trim()) {
      setError("Serial number is required.");
      return;
    }

    if (!deviceTypeId.trim()) {
      setError("DeviceTypeId is required (GUID).");
      return;
    }

    setLoading(true);
    try {
      const id = await createDevice({
        serialNumber: serialNumber.trim(),
        deviceTypeId: deviceTypeId.trim(),
        status: status.trim(),
        currentLocationId: locationId.trim() ? locationId.trim() : null,
      });

      setCreatedId(String(id).replaceAll('"', ""));
      setSerialNumber("");
      setDeviceTypeId("3fa85f64-5717-4562-b3fc-2c963f66afa6");
      setStatus("New");
      setLocationId("");
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
          <label>DeviceTypeId *</label>
          <input
            value={deviceTypeId}
            onChange={(e) => setDeviceTypeId(e.target.value)}
            placeholder="GUID"
            style={{ width: "100%", padding: 8 }}
          />
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
          <input
            value={locationId}
            onChange={(e) => setLocationId(e.target.value)}
            placeholder="GUID"
            style={{ width: "100%", padding: 8 }}
          />
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
