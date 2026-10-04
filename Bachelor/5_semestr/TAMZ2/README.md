# MoldApp

Krátký popis
Aplikace pro Android určená ke správě montáží forem pří výrobě pěn ve společnosi Hzundai Transys. Implementuje volání API pro získávání dat o montážích, uživatelské rozhraní v Kotlinu (Jetpack Compose) a podporu skenování pomocí dodaného Android Scanner SDK od honeweelu.

## Hlavní funkce

1. Čtení a zápis RFID
- Automatické načtení RFID kódu (přes LocalScanner / připojený skener).
- Validace načtených kódů.
- Simulace zápisu pro testovací účely.
- Varianta zápisu carrieru, mold ID nebo repair dat.

2. Správa Carrierů
- Vyhledávání carrierů v databázi přes Mold API.
- Výběr z načteného seznamu (ExposedDropdownMenu).
- Zobrazení detailních informací.
- Odeslání “mount” operace na API (připojení carrieru k formě).

3. Správa forem (Molds)
- Vyhledání formy podle RFID.
- Načtení informací o formě z API.
- Zobrazení historie montáží / oprav (pokud je API podporuje).
- Odeslání příkazu na API pro zápis informací o formě.

4. Režim oprav (Repair Mode)
- Výběr typu opravy z API.
- Vyplnění doplňujících informací.
- Zapsání operace do backend systému.

5. Kontrola připojení
- Real-time detekce ONLINE/OFFLINE stavu.
- Zpracování chyb API a čitelné chybové hlášky pro uživatele.

6. Lokalizace
- Podpora více jazyků přes strings.xml.
- Přepnutí mezi EN / CZ.

7. Moderní UI (Jetpack Compose)
- Material 3 design.
- Přehledná navigace.
- Jednoduché formuláře.
- Dropdowny, validační pop-upy, dialogy.


## Strukturované vrstvy projektu:
- ui/ — obrazovky
- business/ — API repository, modely
- business/localdata/ — local storage
- business/scanners - komunikace se scannerem
- components/ — sdílené komponenty (ButtonMenu, dialogy…)
- app/arr - SDK pro Honeywell CT40