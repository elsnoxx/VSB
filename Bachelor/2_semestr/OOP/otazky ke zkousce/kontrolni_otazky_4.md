# Kontrolní otázky

## Jaký je rozdíl mezi funkční a objektovou dekompozicí programu?
- Funční
    - Co bude systém dělat
    - Obsahuje sadu funkcí
- Objektová
    - Kdo bude funkčnost zajišťovat
    - Jedná se o sadu objektů

## Proč preferujeme objektovou dekompozici a jaké jsou hlavní problémy funkční dekompozice?
- Problémy funkční:
    - Špatná rozšířítelnost
    - Neumí to opakovanou použitelnost
    - Neumí to kombinovatelnost
- Proč?
    - Z důvodu, že tyto problémy neobsahuje
    - Objektová je časově stabilnější 
    - Lépe použitelná v budoucnosti

## Za jakých podmínek můžeme považovat třídu za objekt a jak to implementovat v C++?
- Když má třída v sobě nějaká třídní (static) data/metody
- Když je splněno zapouzdření
- Pajdova odpověď:
    - Třídu považujeme za objekt v případě že obsahuje getry a setry.
    - Neboli, dá se s daným objektem manipulovat
    - Implementujeme pomocí public častí třídy

## Vysvětlete rozdíl mezi členskými položkami třídy a instance a popište jejich dostupnost.
- K položkám třídy má přístup jakákoli instance třídy a dá se k nim dostat i bez vytvoření instance
- Položky instance se mezi instancemi nesdílí

## Jak můžeme v C++ důsledně odlišovat práci s členskými položkami tříd a instancí?
- Konvence pro práci s daty tříd
    - Instanční data/metody: voláme instanci a `->`
    - Třídní data/metody: voláme třídu a `::`

## Potřebuje třída v roli objektu konstruktor resp. destruktor a proč?
- Nepotřebuje v kažém případě
- Konstruktor potřebuje v případě, kdy neobsahuje static hodnoty
    - Static hodnoty se musí inicializovat zvlášť