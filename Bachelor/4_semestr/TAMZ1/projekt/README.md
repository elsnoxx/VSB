# Popis projektu – Mobilní aplikace & API server

## Mobilní aplikace (Ionic + React)

Aplikace obsahuje tyto klíčové funkce a moduly:

- **Práce s GPS:**  
  Získávání aktuální polohy uživatele pro zobrazení na mapě, hledání kešek v okolí a ukládání souřadnic při zakládání nové kešky. Najdete v souborech:  
  - [`src/components/CurrentPosition.tsx`](caching-app/src/components/CurrentPosition.tsx) – výpočet azimutu a zobrazení směru 

- **Import a export dat:**  
  Možnost importovat/exportovat seznam kešek a nálezů ve formátu JSON nebo CSV.  
  [`src/components/CsvImportExport.tsx`](caching-app/src/components/CsvImportExport.tsx)

- **Práce s mapou:**  
  Zobrazení kešek na mapě, navigace k vybrané kešce, zobrazení detailů kešky.  
  - [`src/components/MapContainerMultiple.tsx`](caching-app/src/components/MapContainerMultiple.tsx) – vykreslení více kešek na mapě  
  - [`src/pages/Found.tsx`](caching-app/src/pages/Found.tsx) – zobrazení nalezených kešek na mapě  


- **Kompas:**  
  Zobrazení směru k vybrané kešce pomocí kompasu zařízení.
  - [`src/components/CompassComponent.tsx`](caching-app/src/components/CompassComponent.tsx) – vizuální kompas   

- **Lokální databáze:**  
  Ukládání kešek, nálezů a uživatelských dat do localStorage pro offline použití.

- **Cacheování paměti:**  
  Dočasné ukládání dat z API pro rychlejší načítání a úsporu datového přenosu.

- **Kontrola dostupnosti serveru:**  
  Automatická kontrola, zda je backend server dostupný, případně upozornění uživatele na offline režim.

- **Správa uživatelského účtu:**  
  Přihlášení se pod uživatelským jménem a uložení do localStorage.


---

## API server (ASP.NET Core)

API server slouží jako **centrální bod pro správu a sdílení dat** mezi uživateli a aplikací. Nabízí tyto možnosti:

- **Správa kešek:**  
  - Získání seznamu všech kešek (`GET /api/Caching`)
  - Detail kešky podle názvu (`GET /api/Caching/detail/{name}`)
  - Vyhledávání kešek v okolí (`GET /api/Caching/nearby?lat=...&lng=...&radius=...`)
  - Přidání záznamu o nálezu (`POST /api/Caching/found`)
  - Export kešek jako JSON (`GET /api/Caching/file`)
  - Stažení všech kešek jako soubor (`GET /api/Caching/download`)

- **Kontrola dostupnosti serveru:**  
  - Endpoint pro health check (`GET /api/health`)
---

## Shrnutí

- **Mobilní aplikace** umožňuje uživateli pohodlně vyhledávat, zakládat a nacházet kešky, pracovat s mapou, kompasem, importovat/exportovat data a používat aplikaci i offline.
- **API server** poskytuje rozhraní pro správu dat, synchronizaci a komunikaci mezi uživateli a aplikací.