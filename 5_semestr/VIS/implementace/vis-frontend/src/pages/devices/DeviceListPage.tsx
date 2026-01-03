import React, { useEffect, useState } from "react";
import { getDevices, deleteDevice } from "../../api/devicesApi";
import type { DeviceRow } from "../../api/types";
import { useNavigate } from "react-router-dom";

export function DeviceListPage() {
  const [items, setItems] = useState<DeviceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const navigate = useNavigate();

  useEffect(() => {
    const ac = new AbortController();

    (async () => {
      try {
        setLoading(true);
        setError(null);

        const data = await getDevices(ac.signal);
        setItems(data);
      } catch (e: unknown) {
        if (e instanceof DOMException && e.name === "AbortError") return;
        if (e instanceof Error && e.message.includes("aborted")) return;
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    })();

    return () => ac.abort();
  }, []);

  function onCreateClick() {
    navigate("/devices/create");
  }

  async function onDeleteClick(e: React.MouseEvent, id: string) {
    e.stopPropagation();

    const ok = window.confirm("Do you really want to delete this device?");
    if (!ok) return;

    setDeletingId(id);
    setError(null);

    try {
      await deleteDevice(id);
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
        }}
      >
        <h2 style={{ margin: 0 }}>Devices</h2>

        <button onClick={onCreateClick} style={{ padding: "8px 12px" }}>
          New Device
        </button>
      </div>

      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {!loading && !error && (
        <table style={{ borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr>
              <th style={{ padding: 8, borderBottom: "1px solid #ccc" }}>Serial</th>
              <th style={{ padding: 8, borderBottom: "1px solid #ccc" }}>Status</th>
              <th style={{ padding: 8, borderBottom: "1px solid #ccc" }}>Location</th>
              <th style={{ padding: 8, borderBottom: "1px solid #ccc" }}>Created</th>
              <th style={{ padding: 8, borderBottom: "1px solid #ccc", width: 120 }}>Actions</th>
            </tr>
          </thead>

          <tbody>
            {items.map((d) => (
              <tr
                key={d.id}
                onClick={() => navigate(`/devices/${d.id}`)}
                style={{ cursor: "pointer" }}
              >
                <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>{d.serialNumber}</td>
                <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>{d.status}</td>
                <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>
                  {d.currentLocationId ?? "-"}
                </td>
                <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>
                  {new Date(d.createdAtUtc).toLocaleString()}
                </td>
                <td style={{ padding: 8, borderBottom: "1px solid #eee" }}>
                  <button
                    onClick={(e) => onDeleteClick(e, d.id)}
                    disabled={deletingId === d.id}
                    style={{ padding: "6px 10px" }}
                  >
                    {deletingId === d.id ? "Deleting…" : "Delete"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
