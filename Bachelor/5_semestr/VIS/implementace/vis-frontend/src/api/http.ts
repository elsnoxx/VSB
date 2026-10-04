async function parseError(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  return text || `${res.status} ${res.statusText}`;
}

export async function apiGet<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { method: "GET", signal });
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as T;
}

export async function apiPost<TReq, TRes>(url: string, body: TReq): Promise<TRes> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) throw new Error(await parseError(res));

  // tvoje API vrací Guid (string) nebo objekt – podle implementace
  const text = await res.text();
  try {
    return JSON.parse(text) as TRes;
  } catch {
    // když backend vrátí jen "guid-string" bez JSON
    return text as unknown as TRes;
  }
}
