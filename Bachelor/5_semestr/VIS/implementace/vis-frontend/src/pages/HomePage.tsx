import React from "react";

export function HomePage() {
  return (
    <div style={{ maxWidth: 800 }}>
      <h2>Evidence IT zařízení</h2>

      <p>
        Tato aplikace slouží jako jednoduchý informační systém pro evidenci IT zařízení
        a jejich přiřazení k jednotlivým lokacím v rámci organizace.
      </p>

      <p>
        Systém umožňuje správu hardwarových zařízení (např. notebooky, monitory, servery),
        jejich jednoznačnou identifikaci pomocí sériových čísel a evidenci jejich aktuálního
        umístění.
      </p>

      <h3>Hlavní funkce systému</h3>
      <ul>
        <li>
          <strong>Správa zařízení (UC7)</strong> – přidávání nových zařízení, prohlížení seznamu
          a zobrazení detailu konkrétního zařízení.
        </li>
        <li>
          <strong>Přiřazení zařízení k lokaci (UC10)</strong> – evidence fyzického umístění
          zařízení na konkrétní lokaci.
        </li>
        <li>
          <strong>Správa lokací (UC15)</strong> – vytváření a údržba lokací, ke kterým mohou být
          zařízení přiřazena.
        </li>
      </ul>

      <p>
        Aplikace je navržena jako webový informační systém s odděleným frontendem (React)
        a backendem (ASP.NET Core Web API) a demonstruje použití základních návrhových vzorů
        v architektuře informačních systémů.
      </p>
    </div>
  );
}
