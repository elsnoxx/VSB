# Kontrolní otázky

## Co je čistě virtuální metoda?
- Metoda, která má pouze deklaraci, nemá definici

## Kdy je vhodné použít čistě virtuální metodu? Uveďte příklad.
- Pro správný návrh programu
- Ideální použití jako vzor pro dědičnost potomky

## Co je abstraktní třída?
- Třída, která má alespoň jednu čistě virtuální metodu
- Abstraktní proto, že nemůžeme vytvořit její instanci (protože čistě viruální metoda má sice deklaraci, ale nemá definici/implementaci)
- Může, ale nemusí, mít členské proměnné a implementované metody

## Kdy je vhodné použít abstraktní třídu? Uveďte příklad.
- Když budeme v programu mít dědičnost a budeme chtít předem určit vzor pro potomky

## Má abstraktní třída konstruktor a destruktor? A proč?
- Má konstruktor a destruktor pro potomka

## Může mít abstraktní třída členská data a funkce (metody)?
- Může, ale nemusí, mít členské proměnné a implementované metody

## Co je čistě abstraktní třída?
- Třída, jejíž všechny metody jsou čistě virtuální
- Slouží jako „prázdný“ vzor pro dědičnost
- Deklaruje, ale nedefinuje, budoucí společné chování potomků

## Co je vícenásobná dědičnost?
- Potomek může dědit z více tříd
- Musí se implementovat opatrně, může být nebezpečná a špatně pochopitelná

## Kdy není vhodné použít vícenásobnou dědičnost? Uveďte příklad.
- Když předci mají společný stav nebo chování
- Příklad: když se dědičnost rozdělí a znovu zase sejde

## Kdy je možné použít vícenásobnou dědičnost? Uveďte příklad.
- Potřebujeme-li, aby objekty reprezentovaly v různých situacích různé abstrakce
- Každý potomek je jedinečný a specifický typ předka

## Jaké problémy mohou nastat při použití vícenásobné dědičnosti? Uveďte příklad
- Konflikty jmen
    - Předek může mít stejně pojmenované členské položky (proměnné nebo metody)
    - Dá se vyřešit různými způsoby
- Opakovaná dědičnost
    - Musí se ohlídat, aby se vícekrát nedědilo ze stejné třídy

## Co je opakovaná dědičnost? Uveďte příklady.
- Když se v návrhu stane, že se z jedné třídy dědí vícekrát

## Proč můžeme potřebovat vícenásobnou dědičnost? S čím to souvisí?
- Potřebujeme-li, aby objekty reprezentovaly v různých situacích různé abstrakce
    - Nejlépe dost různé na to, aby nehrozily konflikty jmen
    - Souvisí se zastupitelností předka potomkem
- Ideální je, když jsou předci čistě abstraktní třídy (bez dat)
    - Pak je to totéž jako „interface“