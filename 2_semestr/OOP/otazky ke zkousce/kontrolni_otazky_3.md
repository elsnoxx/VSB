# Kontrolní otázky

## Vysvětlete, jak vznikají objekty třídy, pojem konstruktor a principy práce s ním v C++.
- Objekty vznikají jako instance třídy
- Chování při vzniku instance je definováno konstruktorem
- Konstruktor je metoda bez návratového typu a jeho jediným úkolem je vytvořit a  definovat hodnoty atributů objektu
- Je volán automaticky při vzniku nové instance
- Může jich být definováno více (přetěžování metod) - rozlišeny podle vstupních parametrů

## Vysvětlete, jak zanikají objekty třídy, pojem destruktor a principy práce s ním v C++.
- Destruktor specifikuje chování při zániku instance
- Objekty mohou zanikat více způsoby: 
    - Destruktor voláme sami v bloku programu
    - Destruktor se zavolá automaticky při ukončení programu
- V případě, že deklarujeme novou instanci třídy dynamicky, je destruktor volán při volání operátoru delete
- Chování destruktoru můžeme definovat stejně jako konstruktoru

## Vysvětlete rozdíl mezi statickou a dynamickou deklarací objektů v C++.
- Při **statické** deklaraci objektu je celý objekt uložen v zásobníku
- Při **dynamické** alokaci je v zásobníku uložen pouze ukazatel na adresu v haldě, na níž je objekt uložen

## Jak se dá postupovat, pokud chceme v zadání programu nalézt třídy, jejich metody a datové členy?
- Hledáme v zadání podstatná jména a slovesa

## Kdy a proč potřebujeme použit více konstruktorů jedné třídy?
- Jedná se o vlastnost polymorfismu
- Přetížený konstruktor poznáme tak, že se liší počtem nebo typem parametrů
- Více konstruktorů můžeme použít například když je více způsobů, jak takový objekt vytvořit (viz Account)

## Kdy potřebujeme deklarovat a definovat destruktor?
- Pokud deklarujeme instanci třídy dynamicky

## Co jsou výchozí konstruktory a destruktory a k čemu je potřebujeme?
- Jedná se o konstruktor s prázdným tělem programu 
- Je volán v případě, kdy není definován vlastní

## Jaké typy metod obvykle musíme deklarovat a definovat?
- Getry a Setry

## Co jsou objektové kompozice a k čemu jsou dobré?
- Definujeme třídu využívající jinou třídu, avšak nejedná se o dědičnost
- Například Bank má v sobě seznam Account a seznam Client