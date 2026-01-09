import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createDeviceType } from "../../api/deviceTypeApi";

export function DeviceTypeCreatePage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!name.trim()) {
      setError("Name is required.");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await createDeviceType({ name: name.trim(), description: description.trim() || undefined });
      navigate(-1);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div>
      <button onClick={() => navigate(-1)} style={{ marginBottom: 12 }}>← Back</button>
      <h2>Create device type</h2>

      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      <form onSubmit={onSubmit} style={{ maxWidth: 600 }}>
        <div style={{ marginBottom: 12 }}>
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} style={{ display: "block", width: "100%", padding: 8 }} />
          </label>
        </div>

        <div style={{ marginBottom: 12 }}>
          <label>
            Description
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} style={{ display: "block", width: "100%", padding: 8 }} />
          </label>
        </div>

        <div>
          <button type="submit" disabled={saving} style={{ padding: "8px 12px" }}>{saving ? "Saving…" : "Create"}</button>
        </div>
      </form>
    </div>
  );
}