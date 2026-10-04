
### Cvičení 3

 1. Vytvořte konzolovou aplikaci, která bude po spuštění vyžadovat zadání čísla. Číslo přečtěte (jako string).
 2. Implementujte vlastní metodu **ParseInt**, které předáte textový řetězec a ona vrátí číslo typu int, které tomuto řetězci odpovídá. Při chybném vstupu vraťte -1.
 3.  Implementujte vlastní metodu **ParseIntOrNull**, které předáte textový řetězec a ona vrátí číslo typu int, které tomuto řetězci odpovídá. Při chybném vstupu vraťte hodnotu null.
 4. Implementujte vlastní metodu **TryParseInt**, které předáte textový řetězec a ona vrátí true, nebo false podle toho zdali se parsování čísla podařilo, nebo ne. Výsledek parsování vraťte pomocí výstupního parametru této metody.
 5. Implementujte druhou vlastní metodu **TryParseInt**, které předáte textový řetězec a nastavení reprezentované vlastním výčtovým typem **ParseIntOption**. Metoda vrátí true, nebo false podle toho zdali se parsování čísla podařilo, nebo ne. Výsledek parsování vraťte pomocí výstupního parametru této metody. V rámci výčtového typu **ParseIntOption**, definujte  4 možné nastavení "NONE", "ALLOW_WHITESPACES", "ALLOW_NEGATIVE_NUMBERS" a "IGNORE_INVALID_CARACTERS". Tato nastavení budou modifikovat chování dané metody (povolí mezery/bílé znaky mezi čísly, povolí znaménko mínus a umožní přeskočení všech znaků, které nejsou čísly). Jednotlivá nastavení bude možné kombinovat.