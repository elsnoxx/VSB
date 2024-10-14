### Cvičení 4

 1. Vytvořte konzolovou aplikaci a knihovnu tříd s názvem "Database". Veškerý kód aplikace, pokud nebude uvedeno jinak, bude součástí této knihovny.
 2. Vytvořte vlastní výčtový typ názvem **GenderEnum**. Tento typ bude moci nabývat hodnota **MALE** a **FEMALE**.
 3. Vytvořte třídu **Person**, která bude reprezentovat jednu osobu. Osoba bude mít jméno (string), věk (int) a pohlaví (GenderEnum). Tyto vlastnosti implementujte pomocí properties (vlastností). Věk a pohlavní nebudou povinné údaje. 
 4. Při nastavení věku kontrolujte zdali je v rozmezí 0 až 150. Pokud není, nepovolte jeho nastavení (věk zůstane nevyplněn).
 5. Osoba bude mít navíc vlastnost, která udává zdali je dospělá, nebo ne (bool). Hodnota této vlastnosti bude automaticky vyhodnocena na základě věku dané osoby.
 6. Implementujte vlastní logiku pro převod třídy na text (metoda ToString). V rámci převodu na text vraťte informace o dané osobě v uživatelsky přívětivé formě. Zkuste využít třídy StringBuilder.
 7. Vytvořte třídu **PopulationDatabase**. Tato třída bude umožňovat přidávání přidávání přidávání nových osob do databáze (metoda **Add**). Jednotlivé osoby budou interně (v rámci této třídy) uloženy v rámci pole.
 8. Vytvořte vlastnost **Count**, která bude vracet aktuální počet osob v databázi.
 9. Vytvořte vlastnost **AdultCount**, která bude vracet počet osob, které jsou dospělé.
 10. Vytvořte metodu **GetAverageAge**, která bude vracet průměrný věk osob. Ošetřete situaci, kdy je databáze prázdná, nebo v ní není žádná osoba s vyplněným věkem.
 11. Implementujte vlastní logiku pro převod třídy na text (metoda ToString). V rámci převodu na text vraťte informace o počtu osob, počtu dospělých osob, průměrném věku a seznam těchto osob.
 12. Otestujte vaši implementaci v rámci konzolové aplikace (v metodě Main).
 13. V rámci konzolové aplikace implementujte logiku umožňující zadání příkazu uživatelem (**print**, **add** a **exit**). Na základě zvolené akce se následně spustí potřebná logika (vypsání obsahu databáze, přidání nové osoby, nebo ukončení aplikace). V případě přidání nové osoby bude uživatel vyzván k vyplnění jednotlivých vlastností (nepovinné vlastnosti nebude nutné vyplnit).
