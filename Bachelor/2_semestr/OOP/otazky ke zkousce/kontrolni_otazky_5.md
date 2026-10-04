# Kontrolní otázky

## Které dva klíčové požadavky řešíme pomocí dědičnosti?
- Znovu-použitelnost, Rozšiřitelnost, nechcem opisovat..

## Jaké návrhové požadavky máme na použití tříd (co s nimi můžeme dělat)?
- Rozsireni, Pozmeneni, kombinovani s jinyma tridama z objektů jiné třídy.

## Jaký je rozdíl mezi dědičností a skládáním? Co mají společného?
- Skládáním docílíme toho, že objekt jedné třídy je složen z vícero podtříd
- Dědičností docílíme toho, že nová třída je rozšířením nebo speciálním případem existující třídy (nebo více tříd).
- Instance třídy potomka obsahuje vše, co má instance třídy předka.

## V jakých rolích vystupují třídy v dědičnosti? Použijte správnou terminologii.
- Předek – potomek
- Rodič – dcera
- dasdsadas - pod (sub) třída

## Vysvětlete v jakém obecném vztahu je třída, ze které se dědí, se třídou, která dědí.
- Předek definuje společné chování všech svých potomků.
- Potomci mohou toto chování rozšířit či pozměnit.

## Co všechno se dědí, co ne a proč?
- Dědí se vše

## Co rozumíme jednoduchou dědičností a jak s tím souvisí hierarchie tříd v dědičnosti?
- Každý potomek má právě jednoho předka.
- Předek může mít více potomků.
- V případě jednoduché dědičnosti je touto hierarchií strom.

## Co je Liskové substituční princip a jak se projevuje v dědičnosti?
- Potomek může vždy zastoupit předka…
- … a to proto, že mají společné chování.

## V jakém pořadí se volají a vykonávají konstruktory při použití dědičnosti?
- 1. Volání konstruktoru objektu.
- 2. Volání konstruktoru předka.
- 3. Vykonání konstruktoru předka.
- 4. Vykonání konstruktoru objektu.