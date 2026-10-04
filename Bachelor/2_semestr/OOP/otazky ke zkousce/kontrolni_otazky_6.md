# Kontrolní otázky

## Co rozumíme paradoxem specializace a rozšíření?
- Vztah dědičnosti je vztahem obecný - speciální
- Potomek je tedy speciálním případem předka
- Paradoxem je, že při rozšíření toho potomek vždy umí více než jakýkoli jeho předek

## Uveďte správné a špatné příklady vztahu "generalizace-specializace".
- Potřeba rozšíření sama o sobě není dostačující pro použití dědičnosti
- Špatné: bod jako předek a kružnice jako potomek - kružnice není *speciálním* případem bodu
- Dobré: obdélník jako předek a čtverec jako potomek

## Co rozumíme v dědičnosti změnou chování?
- Pokud je chování deklarováno v předkovi, můžeme ho v potomkovi deklarovat znovu
- Existuje pak více metod stejného jména, kde metody mají různé chování závislé na tom, na jaký objekt je voláme

## Co rozumíme přetížením? Jedná se o rozšíření nebo změnu chování?
- Přetížením rozumíme situaci, kdy daná metoda má stejné jméno, ale má jiné parametry nebo jejich typy
- Typicky konstruktory
- Jedná se o rozšíření

## Uveďte různé typy přetížení.
- Jméno metody zůstává stejné
- Jiný počet parametrů
- Jiné datové typy parametrů
- Jiná návratová hodnota (ne v C++)

## Co rozumíme překrytím? Jedná se o rozšíření nebo změnu chování?
- Překrytím rozumíme situaci, kdy metoda potomka má stejnou deklaraci, jako metoda předka (stejnou signaturu)
- Potomek dědí i metodu předka. Má tedy dvě metody se stejnou deklarací
- Například metoda na výběr peněz z různých typů účtů v bance
- Jde o změnu chování

## Jaký princip porušujeme, použijeme-li „protected“ a proč?
- Zapouzdření
- Private část předka už není soukromá protože je přístupná i potomkům

## Jaký problém přináší potřeba změny chování v dědičnosti?
- Pokud je chování deklarováno v předkovi, můžeme ho v potomkovi deklarovat znovu
- Existuje pak více metod stejného jména

## Popište, jak se prakticky projevuje různá míra přístupu k položkám třídy.
- Public: přístupná všem
- Protected: přístupná třídě a jejím potomkům
- Private: přístupná jen třídě samotné

## Jak se použití „protected“ projeví ve vztahu předka a potomka?
- Potomek má přístup k určitým hodnotám nebo metodám v private části předka