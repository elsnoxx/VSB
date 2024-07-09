# Kontrolní otázky

## Co je hlavním motivem pro vývoj programovacího paradigmatu od imperativního k objektovému?
- Paradigma:
    - Styl, kterým se programuje
    - Můžou to být prvky programu (objekty, funkce, proměnné) nebo kroky výpočtů (přiřazení, vyhodnocení, tok dat) 
- Motiv:
    - Modularita
    - Efektivita

## Co je imperativní programování?
- Popisuje, jak se věci řeší
- Běžně užívané jazyky (C, *Karel*)

## Co je modulární programování?
- Návrh shora dolů
- Velký projekt rozdělí do menších okruhů
- Moduly (bubliny) jsou rozděleny podle funkčnosti, propojeny na základě závislostí a rozkouskovány na další menší podmoduly

## Jaké jsou hlavní faktory kvality software?
- Vnitřní faktory jsou skryty uživateli
- Vnější faktory popisují chování navenek
    - Správnost
        - Míra toho, jak moc program/algoritmus dělá to, co má
    - Robustnost
        - Blbuvzdornost
        - V podstatě obsahuje správnost
        - Kontroluje správné vstupy, správné chování programu v hraničních situacích
    - Rychlost
    - Rozšiřitelnost
    - Použitelnost
    - Kompatibilita

## Co je pochopitelnost modulu? Uveďte příklad.
- Všechny úlohy, které modul provádí musí být jasně definované a zřejmé
- Příklad: program je jasný na pochopení i pro člověka, který ho neprogramoval

## Co je samostatnost modulu? Uveďte příklad.
- Modul musí mít co nejmenší počet závislostí
- Ideálně bychom měli být schopni modul z programu vyjmout a použít v jiném tak, aby všechno fungovalo
- Příklad: soubor funkcí provádějící jisté výpočty můžu použít v libovolném programu

## Co je kombinovatelnost modulu? Uveďte příklad.
- Moduly musí být navzájem kombinovatelné

## Co je zapouzdření modulu? Uveďte příklad.
- Modul si hlídá, ke kterým hodnotám mají ostatní moduly přístup a které jsou soukromé
- Příklad: class KeyValue neposkytuje přímo své hodnoty ale má napsané funkce, které je vrací

## Co je explicitní rozhraní modulu? Uveďte příklad.
- Modul má konkrétně a detailně popsané, jaké předpoklady (vstupy, závislosti) potřebuje pro své správné fungování
- Příklad: funkce násobení nepřijímá jiné vstupy než čísla

## Co je syntaktická podpora modularity?
- Ze zápisu musí být zřejmé, kde začíná a kde končí zápis modulu

## Co je pět kritérií pro dobrou modularitu?
- Dekomponovatelnost
- Kombinovatelnost
- Pochopitelnost
- Kontinuita
- Ochrana

## Co se rozumí pěti pravidly zajišťující dobrou modularitu?
- Přímé mapování
- Pár rozhraní
- Malá rozhraní (weak coupling)
- Explicitní rozhraní
- Skrývání informací

## Popište jednotlivá kritéria dobré modularity. Uveďte příklady.
- Dekomponovatelnost
    - Program dobře rozděluje problém na menší podproblémy
    - Jednotlivé moduly řeší malé problémy a jsou samostatné
- Kombinovatelnost
    - Moduly se dají používat spolu i zvlášť
- Pochopitelnost
    - Člověk dokáže modul jednoduše pochopit bez toho, aby musel rozumět ostatním
- Kontinuita
    - Při malé změně zadání se bude muset změnit jen malé množství modulů
- Ochrana
    - Chyba v jednom modulu se projeví maximálně v pár sousedících modulech a neshodí celý systém

## Popište jednotlivá pravidla pro dobrou modularitu. Uveďte příklady.
- Přímé mapování
    - Vytvořené moduly musí být kompatibilní s moduly definovanými v modelu řešení problému
    - Nesmí znemožnit funkčnost už vytvořeného programu
- Pár rozhraní
    - Moduly by měly komunikovat s minimálním množstvím jiných modulů
- Malá rozhraní (weak coupling)
    - Pokud už moduly musí komunikovat, měly by si vyměnit jen ty nejnutnější informace
- Explicitní rozhraní
    - Musíme jasně vidět, kdy dva moduly komunikují
- Skrývání informací
    - Autor modulu se musí rozhodnout, které informace poskytne autorům jiných modulů

## K čemu je konstruktor? Uveďte příklad.
- Zapíše data do paměti
- Spouští se při deklaraci objektu
- Naplňuje objekt hodnotami

## K čemu je destruktor, kdy ho potřebujeme a kdy ne? Uveďte příklad.
- Čistí paměť
- Není potřeba, když jsou data objektu statická