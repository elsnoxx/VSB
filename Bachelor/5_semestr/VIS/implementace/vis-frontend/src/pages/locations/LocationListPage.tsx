import React, { useEffect, useState } from "react";
import type { LocationRow } from "../../api/types";
import { getLocations, deleteLocation } from "../../api/locationsApi";
import { useNavigate } from "react-router-dom";

export function LocationListPage() {
  const [items, setItems] = useState<LocationRow[]>([]);
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

        const data = await getLocations(ac.signal);
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
    navigate("/locations/create");
  }

  async function onDeleteClick(id: string) {
    const ok = window.confirm("Do you really want to delete this location?");
    if (!ok) return;

    setDeletingId(id);
    setError(null);

    try {
      await deleteLocation(id);
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
        <h2 style={{ margin: 0 }}>Locations</h2>

        <button onClick={onCreateClick} style={{ padding: "8px 12px" }}>
          New location
        </button>
      </div>

      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {!loading && !error && (
        <table style={{ borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: 8 }}>Name</th>
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: 8 }}>Created</th>
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: 8, width: 120 }}>
                Actions
              </th>
            </tr>
          </thead>

          <tbody>
            {items.map((l) => (
              <tr key={l.id}>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>{l.name}</td>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>
                  {new Date(l.createdAtUtc).toLocaleString()}
                </td>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>
                  <button
                    onClick={() => onDeleteClick(l.id)}
                    disabled={deletingId === l.id}
                    style={{ padding: "6px 10px" }}
                  >
                    {deletingId === l.id ? "Deleting…" : "Delete"}
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
