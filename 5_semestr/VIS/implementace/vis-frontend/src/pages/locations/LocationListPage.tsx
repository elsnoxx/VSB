import React, { useEffect, useState } from "react";
import type { LocationRow } from "../../api/types";
import { getLocations, createLocation } from "../../api/locationsApi"; 
import { useNavigate } from "react-router-dom";

export function LocationListPage() {
  const [items, setItems] = useState<LocationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();

    (async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getLocations(ac.signal);
        setItems(data);
      } catch (e: unknown) {
        console.error("getLocations failed:", e);
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    })();

  }, []);

  const navigate = useNavigate();

  function onCreateClick() {
    navigate("/locations/create");
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
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: 8 }}>ParentId</th>
              <th style={{ textAlign: "left", borderBottom: "1px solid #ccc", padding: 8 }}>Created</th>
            </tr>
          </thead>
          <tbody>
            {items.map((l) => (
              <tr key={l.id}>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>{l.name}</td>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>{l.parentId ?? "-"}</td>
                <td style={{ borderBottom: "1px solid #eee", padding: 8 }}>
                  {new Date(l.createdAtUtc).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}