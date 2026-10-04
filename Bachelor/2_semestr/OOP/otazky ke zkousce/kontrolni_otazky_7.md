# Kontrolní otázky

## Jaký je rozdíl mezi shadowing a overriding překrytím? Uveďte příklady
- Shadowing (method hiding)
    - Jde o statické překrytí, kdy nová metoda potomka „zastíní“ metodu předka
    - Dílčí chování objektu tedy odpovídá třídě, v jejíž roli vystupuje
- Overriding
    - Jde o dynamické překrytí, kdy se vždy (i v roli předka) použije metoda potomka, pokud ji má implementovanou
    - Dílčí chování objektu tedy odpovídá třídě, jejíž je tento objekt instancí

## Co rozumíme polymorfismem a s čím to souvisí?
- Polymorfismus je schopnost objektu vystupovat v různých rolích a podle toho se chovat
- Kombinuje své chování s chováním předka, jinak se o skutečný polymorfismus nejedná
- Souvisí to se substitučním principem, tedy se zastupitelností předka potomkem

## Co rozumíme polymorfním přiřazením?
- Když je zdroj přiřazení jiného typu než cíl přiřazení

## Co je časná vazba? Uveďte příklady.
- Když překladač při volání metody vyhodnocuje typ instance již v době překladu
- Časná vazba je výchozí - použije se když neoznačíme metodu jako virtual
- Při volání metody předka Withdraw se zavolá metoda CanWithdraw taky od předka a nezáleží na tom, o jaký objekt se ve skutečnosti jedná

## Co je pozdní vazba? Uveďte příklady.
- Potřebujeme zjistit, kdo metodu žádá, ale až v okamžiku volání
- Když použijeme virtual - Shape má svoji virtual metodu Area ale volat se bude ta metoda, která patří Třídě objektu

## Popište, co je virtuální metoda a její vlastnosti.
- Chceme-li aby se rozhodlo, která překrytá metoda bude volána, až v průběhu programu (overriding), použijeme virtuální metodu
- Dáváme překladači najevo, že si přejeme využít dynamickou nebo také pozdní vazbu (late binding)

## Popište, co je tabulka virtuálních metod a jak funguje.
- Jakmile některou metodu definujeme jako virtuální, překladač přidá ke třídě „neviditelný ukazatel“, který ukazuje do speciální tabulky nazvané tabulka virtuálních metod (VMT).
- Pro každou třídu, která má alespoň jednu virtuální metodu, překladač vytvoří tabulku virtuálních metod
- Tabulka obsahuje ukazatele na virtuální metody
- Tabulka je společná pro všechny instance dané třídy

## Může být konstruktor virtuální? A proč?
- Nemůže
- Před jejich voláním není ještě vytvořen odkaz do VMT

## Může být destruktor virtuální? A proč?
- Může
- Můžeme tak ničit objekty bez znalosti jejich typu

## Kdy mluvíme v C++ o polymorfismu a jak se to projeví v návrhu?
- Polymorfismus je spojen s dědičností
- Nemá smysl mluvit o polymorfismu, pokud nepoužijeme virtuální metody (overriding)
- Jde stále o zastupitelnost předka potomkem

## Co je polymorfní datová struktura a k čemu ji využíváme?
- Struktura která obsahuje objekty různých tříd
- Po takto uložených objektech můžeme požadovat (volat) pouze společné metody předka
- Jak volat ostatní metody objektu vráceného v typu předka? Je nutno přetypovat – je to jedno z omezení polymorfismu.

## Kdy potřebujeme virtuální destruktor? S čím to souvisí?
- Když nevíme typ objektu, který potřebujeme zničit