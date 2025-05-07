docker run --detach --name parkinglot --env MARIADB_ROOT_PASSWORD=myparkinglot -p 3306:3306 mariadb:latest


user----> webapi
pass----> wabapplogin

1. Zobrazení parkovišť na mapě včetně aktuálních počtů volných míst
Chybí: Implementace mapy, která zobrazuje parkoviště s aktuálními počty volných míst.
Řešení:
Použijte knihovnu jako Leaflet.js pro zobrazení mapy.
Přidejte endpoint v API, který vrátí seznam parkovišť s počty volných míst.
Aktualizujte view Index.cshtml nebo jiné relevantní view, aby zobrazovalo mapu.


2. Obsazení parkovacího místa (náhodné přiřazení volného místa)
Chybí: Funkce pro náhodné přiřazení volného parkovacího místa.
Řešení:
Přidejte metodu v ParkingSpaceRepository, která vrátí náhodné volné místo.
Implementujte API endpoint v ParkingSpaceApiController pro obsazení místa.
Aktualizujte UI (např. dialog nebo stránku) pro spuštění obsazení.
Příklad metody pro náhodné volné místo:

3. Ověření stavu při obsazení
Chybí: Kontrola, zda je parkovací místo dostupné před obsazením.
Řešení:
Přidejte validaci v API endpointu pro obsazení místa.
Pokud není žádné volné místo, vraťte chybu.

4. Změna stavu parkovacího místa
Chybí: Omezení změny stavu pouze na "dostupné" a "v údržbě".
Řešení:
Přidejte validaci v API endpointu pro změnu stavu.
Povolené změny:
"dostupné" → "obsazené"
"dostupné" → "v údržbě"
"v údržbě" → "dostupné"

5. Historie stavů parkovacích míst
Chybí: Automatické zaznamenávání historie při změně stavu.
Řešení:
Aktualizujte metodu pro změnu stavu v ParkingSpaceRepository, aby zaznamenávala historii do tabulky StatusHistory.
Příklad:

6. Zákaz úprav ukončených obsazení
Chybí: Kontrola, aby ukončené obsazení nebylo možné upravovat nebo mazat.
Řešení:
Přidejte validaci v API endpointu pro úpravu obsazení.
Pokud je end_time vyplněno, vraťte chybu.

7. Statistika ukončených parkování za poslední měsíc
Chybí: Funkce pro zobrazení statistiky ukončených parkování.
Řešení:
Přidejte metodu v ParkingLotRepository, která vrátí počet ukončených parkování za poslední měsíc pro každé parkoviště.
Implementujte API endpoint pro získání této statistiky.
Aktualizujte UI (např. stránku nebo dashboard) pro zobrazení statistiky.
Příklad metody:

8. UI pro zobrazení historie stavů
Chybí: Stránka nebo dialog pro zobrazení historie stavů parkovacích míst.
Řešení:
Přidejte view (např. History.cshtml) pro zobrazení historie.
Implementujte API endpoint pro získání historie stavů.

9. Testování a validace
Chybí: Testování všech funkcí a validace vstupů.
Řešení:
Otestujte všechny API endpointy a UI.
Přidejte validace na straně klienta i serveru.
