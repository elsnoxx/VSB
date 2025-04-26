# Návod na spuštění a vývoj aplikace

## Frontend (Ionic + React)

### 1. Inicializace a nastavení projektu
- Nainstaluj Node.js a Ionic CLI (pokud ještě nemáš).
- Vytvoř nový projekt v Ionic + React:
  ```bash
  ionic start myApp blank --type=react
  cd myApp
  ```
- Přidej Capacitor pro nativní podporu (pro Android/iOS):
  ```bash
  npx cap add android
  npx cap add ios
  ```

### 2. Instalace potřebných pluginů
- Nainstaluj pluginy pro GPS a notifikace:
  ```bash
  npm install @capacitor/geolocation @capacitor/local-notifications
  npx cap sync
  ```

### 3. Vytvoření služby pro komunikaci s backendem
- Vytvoř složku `services/` a soubor `cacheService.ts`.
- Napiš funkce pro komunikaci s API (GET pro kešky, kontrola verze).

### 4. Implementace kontrolování verze aplikace
- Vytvoř v `App.tsx` nebo na jiném místě funkci pro kontrolu verze.
- Načti verzi z backendu a porovnej ji s aktuální verzí.
- Pokud je k dispozici nová verze, zobraz notifikaci nebo alert.

### 5. Implementace zobrazení kešek
- Vytvoř stránku `MapPage.tsx` pro zobrazení seznamu kešek.
- Použij funkci `getAllCaches()` z `cacheService.ts` pro načítání kešek.
- Zobraz seznam kešek s odkazem na detailní stránku (např. `CacheDetailPage`).

### 6. Implementace notifikací pro novou verzi
- Nainstaluj a použij plugin pro lokální notifikace (`@capacitor/local-notifications`).
- Při zjištění nové verze aplikace spusť notifikaci s odkazem na stáhnutí nové verze.

### 7. Testování aplikace
- Spusť aplikaci na emulátoru nebo fyzickém zařízení:
  ```bash
  npx cap open android   # pro Android
  npx cap open ios       # pro iOS
  ```
- Otestuj funkčnost zobrazení kešek, kontrolu verze a notifikace.

---

## Backend (ASP.NET Core)

### 1. Vytvoření projektu v .NET
- Vytvoř nový projekt typu ASP.NET Core Web API.
- Přidej potřebné balíčky (např. Entity Framework pro práci s databází, pokud používáš DB).

### 2. Vytvoření API pro kešky
- Vytvoř controller `CacheController.cs` pro zpracování požadavků na kešky:
  - `GET /api/caches` pro získání seznamu kešek
  - `GET /api/caches/{id}` pro získání detailu konkrétní kešky.

### 3. Vytvoření API pro kontrolu verze
- Vytvoř controller `VersionController.cs` pro poskytování informací o poslední verzi aplikace:
  - `GET /api/version` vrací informace o nejnovější verzi, URL pro stažení .apk a zprávu pro uživatele.

### 4. Příprava dat (kešky)
- Vytvoř třídu pro model `Cache.cs` (název, popis, lokaci, atd.).
- Naplň server vzorovými daty (buď hardcodovaná nebo načítání z databáze).

### 5. Testování API
- Ověř funkčnost API pomocí nástrojů jako Postman nebo Insomnia:
  - Testuj `/api/caches` pro seznam kešek.
  - Testuj `/api/version` pro verzi aplikace.

### 6. Nasazení serveru
- Nasadit backend na server (např. pomocí IIS, Azure nebo jiného hostingu).
- Zkontroluj, že API endpointy fungují správně.

---

## Distribuce aplikace

### 1. Příprava pro nativní aplikace

**Pro Android:**
```bash
ionic build
npx cap open android
```
V Android Studiu vytvoř APK nebo AAB.

**Pro iOS:**
```bash
ionic build
npx cap open ios
```
V Xcode vytvoř IPA.

### 2. Testování a distribuce
- Otestuj `.apk` nebo `.ipa` na reálném zařízení.
- Pokud je potřeba, nastav způsob distribuce (např. přes vlastní server pro stažení APK).
- Zajisti, že uživatelé dostanou notifikaci o nové verzi a budou ji moci stáhnout.

### 3. Vydání aplikace
- Pokud chceš, můžeš aplikaci publikovat na Google Play nebo App Store:
  - Pro Android musíš vytvořit signed APK nebo AAB a nahrát do Google Play Console.
  - Pro iOS musíš vytvořit signed IPA a nahrát přes Xcode nebo Transporter do App Store Connect.

---

## Závěr

- Zkontroluj celý projekt a otestuj všechny funkce.
- Pokud vše funguje, můžeš aplikaci zveřejnit nebo nasadit na svůj server pro interní použití.