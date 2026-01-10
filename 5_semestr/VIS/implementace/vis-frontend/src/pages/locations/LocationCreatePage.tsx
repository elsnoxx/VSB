import React, { useState } from "react";
import { createLocation } from "../../api/locationsApi";

export function LocationCreatePage() {
  const [name, setName] = useState("");
  const [parentId, setParentId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdId, setCreatedId] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setCreatedId(null);

    const trimmed = name.trim();
    if (!trimmed) {
      setError("Name is required.");
      return;
    }

    setLoading(true);
    try {
      const id = await createLocation({
        name: trimmed,
        parentId: parentId.trim() ? parentId.trim() : null,
      });

      setCreatedId(String(id).replaceAll('"', "")); // kdyby backend vrátil JSON string
      setName("");
      setParentId("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2>Create Location</h2>

      <form onSubmit={onSubmit} style={{ maxWidth: 420 }}>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", marginBottom: 4 }}>Name *</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ width: "100%", padding: 8 }}
            placeholder="e.g. Building A / Office 305"
          />
        </div>

        <button type="submit" disabled={loading} style={{ padding: "8px 12px" }}>
          {loading ? "Saving…" : "Create"}
        </button>
      </form>

      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}
      {createdId && <p style={{ color: "green" }}>Created! New id: {createdId}</p>}
    </div>
  );
}
