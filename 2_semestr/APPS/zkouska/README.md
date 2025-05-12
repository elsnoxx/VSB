# Architektury počítače

### Základní principy fungování počítače

počítač je programován obsahem paměti, instrukce z paměti se vykonávají sekvenčně a každý krok je závislý na předchozím (takový velký state machine)

> IP (instruction pointer)/PC (program counter) - registr obsahující další instrukci

### Výhody a nevýhody von Neumann

#### Výhody

Rozdělení paměti kódu a dat určuje programátor
Stejná instrukce k paměti a datům (instrukce)
Jednodušší výroba (jedna sběrnice)

#### Nevýhody

Možnost vlastního přepsání programu (protože data a paměť jsou vedle sebe)
Pomalé - kvůli jedné sběrnici

### Výhody a nevýhody Harvardské

#### Výhody

Program se nemůže přepsat
Paměti můžou být odlišné technologie a jiné velikosti nejmenší adresovací jednotky
Souběžný přístup k instrukcím a paměti

#### Nevýhody

Složitější výroba (dvě sběrnice)
Nevyužitou paměť nelze využít k uložení instrukcí a naopak

### Podpora paralelismu u obou architektur

Sériové zpracování instrukcí - paralelismus se musí simulovat v OS

### Je lepší mít oddělené data a paměť? Proč ano a proč ne?

Ano - můžou se používat jiné technologie, nepřepíše se program
Ne - dražší na výrobu

### Může fungovat PC bez paměti nebo bez periferií?

Ne

> Neměl by s čím počítat, kam to uložit a jak to zobrazit

### K čemu se v PC využívá dvojková soustava

Reprezentace čísel, instrukcí, adres, znaků... ke všemu

### Zvyšují sběrnice výkon pc?

Ano, je pak možné zvětšit objem přenesených dat

### Je možné aby procesor prováděl instrukce jinak, než sekvenčně?

Ne

> Musí se buď přidat více procesorů/jader nebo simulovat paralelizmus v os

### Jak je v PC organizovaná pamět?

Je rozdělena do buněk stejné velikosti a přistupuje se pomocí offsetu (adresa)

## Technologie výroby číslicových obvodů

### Bipolární technologie

Rychlejší než unipolární, mají větší spotřebu a nižší cenu. Nejsou tak dobře integrované jako uni.

#### DTL (Diode-transistor logic)

Dnes již nepoužíváné. zpoždění 10 - 30 ns

#### TTL (Transistor Transistor Logic)

Základ tvoří tranzistor s vícenásobným emitorem (umožňuje realizovat logické funkce). Zpoždění 20 ns, příkon 10 mW

> idk co tu napsat

#### STTL

Variatna TTL s příznivějším poměrem rychlosti a příkonu.
Využívají se schottkyho diody pro zvýšení rychlosti.
Zpoždění poloviční, ale příkon dvojnásobný

| technologie | zpozdeni (ns) | odber clenu (mW) | zatizitelnost vystupu (mA) | odber vstupu (mA) |
|-------------|---------------|------------------|----------------------------|-------------------|
| TTL | 22 | 10 | 16 | 1,6 |
| HTTL | 13 | 22 | 30 | 3 |
| LTTL | 35 | 2 | 8 | 0\.8 |
| STTL | 5 | 29 | 50 | 5 |
| LSTTL | 20 | 3 | 8 | 0\.8 |
| ASTTL | 1\.8 | 8 | 7 | 0\.7 |
| ALSTTL | 15 | 2 | 4 | 0\.4 |

#### IIL

Velmi rychlá technologie s malým příkonem (méně než 0.05 mW).

#### ECL

?

### Unipolární technologie

Díky unipol. tech. jsou dnes PC, protože umožňují obrovskou integraci.
Pricip - čím menší vodivost látky, tím lépe do ní může elektrické pole vniknout

ChatGPT:

> ELI5: Unipolární tranzistory jsou jako malé vypínače ve věcech jako počítače. Mají speciální kanál, kterým může proudit elektřina. Když pošleme zprávu na jejich bránu, kanál se otevře a elektřina může volně procházet. Když pošleme jinou zprávu, kanál se zavře a elektřina přestane proudit. To nám pomáhá ovládat proud elektřiny v počítači.

#### PMOS a NMOS (Positive/Negative Metal Oxid Semiconductor)

Jsou řízeny elektrickým polem, ne proudem -> nižší spotřeba
Nízká rychlost. Téměř se nepožívá

#### CMOS

Spojení NMOS a PMOS.
Dobrá odolnost proti šumu a nízký příkon.
Nejpoužívanější technologie ze všech.

#### SOI a SOS (Silicon on Insulator/Sapphire)

Skupina technologií s podobným principem - destička safíru (nevodiče). Díky tomu je až 3x menší parazitní kapacita, vyšší spínací rychlosti a 4x větší hustota než CMOS

#### FAMOS a FLOTOX

Technika plovoucího hradla s lavinovou injekcí nosičů.
Nejrozšířenější pro výrobu EPROM.
Základem je MOS tranzistor a řídicí elektroda, která není k ničemu připojena.
Mazání informace se dělá pomocí UV světla, protože díky UV světlu elektroda získá dostatek energie na překonání bariéry.

FLOTOX je modifikace, která se maže pomocí elektřiny - EEPROM

#### CCD (Charge-coupled device)

Není používaný k zesílení, ale k přenosu náboje.
Nepoužívá se u mikropočítačů, používají se spíš v analogové technice - paměti ve snímačích obrazu pro televizi.
Malá spotřeba a malé rozměry

### Rozdíl mezi transistory (uni, bi)

> [zdroj](https://www.mylms.cz/50-rozdil-mezi-bipolarnim-a-unipolarnim-tranzistorem/)

Bipolární tranzistor se skládá ze tří na sobě ležících vrstev, tvořených střídavě polovodičem N a P. Všechny tři vrstvy jsou opatřeny vodivými kontakty. Prostřední vrstva se nazývá Báze (B) vnější pak Kolektor (K/C) a Emitor (E). V bipolárním tranzistoru se na zesilování podílí oby typy nosičů.

Unipolární tranzistory jsou tvořeny křemíkovým monokrystalem s dotací typu N nebo P, který je na obou koncích opatřen vodivými kontakty. Po stranách dráhy mezi Source (S) a Drain (D) jsou 2 propojené plošky tvořící řídící elektrodu Gate (G). V unipolárním tranzistoru se na zesilování podílí pouze jeden typ nosičů

Příklad bipolárních - NPN, PNP
Příklad unipolárních - FET, MOSFET

---

# Komunikace s periferémi

Sběrnice - taková dálnice pro signály

druhy sběrnice - Datová, adresová, řídící

### Adresová sběrnice

Mikroprocesor nastaví adresu na kterou chce zapisovat nebo číst.
Počet bitů sběrnice = počet drátů = počet bitů adresy

Univerzání mikroprocesory mají většinou dvě adresové sběrnice - pro paměť a pro IO

### Řídící

Sběrnice pro nastavení dobrého stavu zařízení.
Příklad signálů: RESET, Memory read, Memory write, IO R/W, READY

### Datová

Např. přenos dat z RAM do mikroprocesoru
Vždy musí být aktivní jenom jeden vysílač (= budič sběrnice)
Třístavové budiče sběrnice - odpojuje sběrnici od zařízení ("zamykání")
Důležité parametry - šírka sběrnice a časování
Šířka sběrnice - kolik bitů lze najednou přenést

Multiplexace sběrnice - posílají se dva balíky dat ve "najednou" - bity jsou aktivní v jinou dobu - určuje řídící sběrnice

Typy sběrnic:

* Vnitřní sběrnice mikroprocesoru
* Vnitřní sběrnice PC - propojuje s jinými prvky PC
* Vnější sběrnice PC - pro připojení dalších zařízení, umožňuje nastavovat prioritu zařízení a měnit vlastnosti podle potřeby za běhu

### Adresový dekodér, připojení zařízení ke sběrnici

DMA se využívá pro přenos velkých dat - levné na výrobu, snadná změna algoritmu
Dva adresové prostory - paměť a IO
Mapování periferií - každá periferie má svůj adresový prostor - rychlejší přístup
Paměťový prostor mívá víc jak jeden čip paměti nebo několik zařízení, proto se používá adresový dekodér.
Adresový dekodér má CS (Chip select) signál pro jednotlivé obvody - připojuje požadované komponenty na sběrnici

### Řízení komunikace

Dva případy zahajování komunikace:

1. Iniciativa programu - např požadavek na vysílání signálu
2. Iniciativa periferie - např uživatelský vstup z klávesnice

V případě, že periferie je aktivování, řeší se zpracování několika způsoby:

* Obvodovým řešením periferie - počítač o tom nemusí vědět - u PC nízké a střední integrace
* Bit flag - Zařízení nastaví flag na  1, PC si ho přečte a začne číst (progamové řízení) - může indikovat stav zařízení (připraven, zaneprázdněn, chyba...) 

  > tento proces se nazývá polling a používá se pro procesy, které nepotřebují okamžité odbavení. pokud si procesor nepřečte data dostatečně rychle, zařízení poté čeká ve smyčce (spin-lock), dokud nebudou data zpracována
* Interrupt - zařízení přeruší procesor, ten vykoná čtení a poté se vrátí k původní práci 

  > v procesoru je tzv. vektor přerušení (interrupt vector table), ve kterém jsou vydefinované adresy funkcí kam má procesor skočit když interrupt nastane. procesor může zjistit odkud přerušení přišlo několika způsoby:
  > 1. zařízení přidá k interruptu identifikační znak
  > 2. procesor se zeptá každého zařízení jestli interrupt vyslalo (to umožňuje priority interruptů)
  > 3. řadič přerušení (také umožňuje nastavit priority)
* DMA - zařízení přímo zapíše do paměti PC svoji zprávu

### Technika IO bran

Zprostředkovává předávání dat mezi PC a periferií

Několik způsobů předávání dat (R/W):

* nepodmíněný - PC/periferie přikáže druhé straně R/W 

  > vyžaduje neustálou připravenost  obou zařízení
* podmínený - nastaví flag na 1, zařízení až bude mít čas tak provede operaci a poté flag opět vynuluje
* buffered - vyžaduje extra registry do kterého zařízení zapíše své data a nastaví flag, že data jsou připravena. poté druhé zařízení začne číst z těchto registrů, vynuluje flag, nastaví ACK flag a tím řekne, že je operace dokončena a řekne druhému zařízení že opět může zapisovat do registru

### DMA (direct memory access)

Zařízení začne psát do paměti (téměr) bez účasti procesoru - nemusí ukončovat svou aktuální práci
Jediná podmínka je volná sběrnice a jediná práce procesoru je zamknout sběrnici pro ostatní zařízení
Struktura DMA bloku:

* registr dat - co má být přesunuto
* registr adresy - kam má být přesunuto
* čítač přesunů - kolik má být přesunuto

Proces konfigurace a používání DMA:

1. CPU nakonfiguruje DMA blok
2. Povolení periferií k DMA a čekání na žádost
3. Po přijetí žádosti k DMA procesor vyšle ACK a uvolní sběrnice
4. zařízení přenáší data a snižuje čítač přenosů až do nuly
5. Periferie ukončí DMA a předá řízení procesoru
6. Procesor pokračuje v práci dokud další zařízení nepožádá o další DMA

### Kanálová architektura

Kanál je programově řízený specializovaný procesor, který je umístěný mezi procesorem a řadičem zařízení a umí se řídit pomocí IO operací. Jeden systém může obsahovat několik IO kanálů.
Tento proces se řízený pomocí master/slave - master rozdává přístup k hlavní paměti. Ke každému kanálu jsou připojená periferní zařízení.

Kanál má také možnost DMA.
Selektorový kanál - jedna periferie
Multiplexní kanál - více periferií na jednom kanálu.

### Žádosti a priority

K řízení systému se používají žádosti - signál, že je zařízení připraveno komunikovat.
Žádostí může být několik - proto se zavádějí priority - podle nich se určuje zpracování žádostí.

Druhy žádostí:

* samostatné 

  > Každé zařízení má dva dráty do procesoru (req/res). Zařízení nastaví req na 1 - chce zapisovat, procesor nastaví res na 1 - může zapisovat na sběrnici.
* Zřetězení 

  > Požadavky jsou na jednom vodiči a požadavky chodí anonymně. Každá žádost obsahuje "id" a procesor potom pošle toto "id" sériově na všechna zařízení a pokud se shoduje s id zařízení, může psát.
* cyklické 

  > podobné jako u zřetězení, ale vodič(e) navíc, který(é) obsahují adresy zařízení

### Kontrolní otázky

Z jakých částí se skládá sběrnice?

> adresová, datová, řídící. vždy několik vodičů (šířka)

Co je adresní dekodér a kdy je potřeba ho použít?

> připojuje zařízení na sběrnici, potřebujeme ho když má systém víc zařízení

Jaký je princip komunikace s periferiemi pomocí IO bran?

> zařízení požádá přes IO bránu o zápis (nebo procesoru přikáže zápis/čtení)

K čemu slouží u komunikace přes IO brány indikátor a jaké jsou jeho výhody?

> značí, že zařízení chce komunikovat. výhody - procesor ne vždy může mít čas požadavek zpracovat hned, zařízení potom musí čekat

Jak probíhá přes dat přes IO bránu s indikátorem?

> zařízení nastaví indikátor na 1, procesor si toho po nějké době všimne a začne ze zařízení číst, čímž se vynuluje indikátor

Rozdíl mezi prorgamově řízenou komunikací a interruptem?

> prog. říz. kom - procesor se musí ptát zařízení jestli něco chce předat (nebo se podívá na nějaký flag, ale musí se o to starat), interrupt - zařízení v podstatě zastaví procesor a přikáže mu aby se mu věnoval

Části DMA

> registr dat, registr adresy, čítač přesunů

Průběh přenosu dat s DMA

> zařízení požádá o DMA, procesor uvolní sběrnici, zažízení zapisuje data a snižuje čítač, jakmile je čítač na nule předá zařízení kontrolu sběrnice procesoru

Výhody DMA vs CPU

> díky dma se procesor může zabývat jinýma věcma mezitím co zařízení zapisuje. přes cpu se musí procesor věnovat pouze přesunu dat a ničemu jinému

### RISC/CISC CPU

Vlastnoti RISC:

* malý instrukční soubor
* instrukce mají stejnou délku, stejný formát a trvají stejně dlouho
* zřetězené zpracování instrukcí
* komunikace s pamětí pouze LOAD a STORE
* vyšší počet registrů
* složitost se přesouvá do kompilátoru

Zpracování instrukcí RISC vs CISC:
CISC procesor se věnuje pouze jedné instrukci - nejdříve ji načte, pak dekoduje...

RISC na druhou stranu tento proces zrychlí - načte první, dekoduje první a zároveň načítá druhou...

#### Problémy s zřetězeným zpracováním

Některá instrukce může potřebovat výsledek předchozí instrukce, ale ten ještě nemusí být uložený v registru/paměti
Větve můžou procesor rozhodit, protože může načíst instrukce z druhé větve a potom musí všecho zahodit a načítat první

#### Branch prediction

Moderní procesory umí částečně předvídat výsledek ifu, aby načetl správne instrukce. Branch prediction se dělí na dva typy: statická a dynamická.

Statická znamená, že v kódu je zaznačeno, která větev je pravděpodobnější (vkládá programátor nebo optimalizátor)

Dynamická se přispůsobí situaci a přepíše si kus progamu a zaznačí si tam pravděpodobnější větev.

Dvoubitotová predikce: Procesor si zapisuje předchozí výsledky a podle toho určuje další

---

# Monolitické PC

Nenáročné PC integrované v jednom pouzdře (CPU, RAM, (GP)IO)

> Také jako jednočip, mikropočítač, mikrokontrolér, mikročip

Spíš harvardská architektura - paměť a data používají jinou technologii a jiné velikosti adresování a spíš RISC v jednodušší formě.

Dva druhy pamětí

* RWM-RAM *statické paměti realizované klopnými obvody*
* Flash EEPROM

### Organizace paměti

* Pracovní registry - obvykle 1-2, ukládají se výsledky výpočtů
* Univerzální zápisníkové registry - uložení nejčastěji používaných dat
* RWM data - méně používáné data - většinou s nimi nejde pracovat na přímo, ale přes registy Dále: Program counter, stack, stack pointer

### Synchronizace

Builtin oscilátor - není spolehlivý a odchylky můžou být až desítky % (nelze použít u aplikací s potřebou přesného časování)

Externí oscilátory:

* krystal (křemenný výbrus)
* keramický rezonátor
* obvod LC
* obvod RC

Krystal, keramický rezonátor - velmi stabilní, ale dražší

RC oscilátory - levné, ale mění frekvenci s teplotou a napětím

Spoustu mikročipů má také stavy: IDLE, SLEEP, HALT, které sníží frekvenci a tím pádem spotřebu

### Reset

Reset je stav počítače. Je garantováno, že pc bude v tomto stavu po spuštění/stisknutí reset tlačítka. Poté se musí opět nastavit registry, procesor a periferie.

### Ochrany

Mechanická ochrana (pouzdro, čip v krabičce...)
Galvanicé stínění - proti elektromagnetickému vlivu
Watchdog - proti chyby v kódu
speciální obvod také hlída pokles napětí

### Interrupt system

Snižuje odezvy na periferie

Začátky funkcí musí být na předem definovaných adresách. Po ukončení funkce přerušení se musí nastavit registry na původní stav.

Programátor musí vědet jak povolit, zakázat a zjistit zdroj požadavku interruptu

### IO

Nejčastěji u mikropočítáčů bývá paralelní brána/port. 4 - 8 jednobitových vývodů, kde lze zapisovat i číst. Programátor si nastaví, které vývody budou vstup a které výstup.

> V případě, že není HW podpora např sériového portu, je nutno tuto funkcionalitu emulovat softwarově - potřeba si hlídat výkon CPU v takových případěch

#### Serial

Efektivní přenos dat na relativně velkou vzdálenost při minimálním počtu vodičů. Hodně nízká přenosová rychlost + střídání všech informací na jednom vodiči.

Komunikace mezi zařízeními - RS232 nebo RS485
Komunikace uvnitř zařízení - I2C

__Sériový synchronní a asynchronní přenos dat__
Jeden z nejstarších způsobů přenosu dat. Téměř každý PC má RS232 linku. Nutno generátor hodinového signálu a jejich synchronizace na obou stranách (=další vodič).

V klidovém stavu je linka v logické 1 a když chce začít přenos, nastaví ji na 0.

#### Čítače a časovače

__Čítač vnějších údálostí__
Registr o N bitech. Lze nastavit kdy se zvyšuje - náběhná nebo sestupná hrana. Při přetečení se notifikuje interrupt v CPU. Lze kdykoliv odpojit a nastavit hodnotu programově
__Časovač__
Skoro stejný jako čítač, ale je inkrementován hodinovým signálem. Obsahuje také prescaler, kterým si programátor může nastavit jemnost časovače

#### ADC

Převádí analogovou hodnotu na digitální. (analog - napětí, proud nebo odpor)

__Komparační__
Porovnání s referenčními hodnotami, rozdělené v určitém poměru (např odporovou děličkou). Díky tomu dostaváme paralelní převodník. Jsou velmi rychlé a používají se ke zpracování signálů.

__ADC s DAC__
Jediný komparátor a proměnný zdroj hodnoty. Dva druhy: sledovací a aproximační.

__Sledovací__ - mění ref. hodnotu a jeden krok nahoru nebo dolů - velmi pomalé a používají se např pro teplotu nebo vlhkost

__Aproximační__ - Vždy se půlí interval a komparátor se podívá jestli je hodnota větší nebo menší (něco jako binary search tree)

__Intergrační ADC__
idk jak napsat tldr, ale funguje to asi takhle: integruje napětí po pevnou dobu, poté čte a integruje napětí opačné polarity stejnou dobu.

Nelze vyrobit jako integrovaný obvod, vždy externí součástka

__Převodníky s časovacím RC článkem__
Pro měření odporu (termistor). Měří se za jak dlouho se nabije/vybije kondenzátor.

#### DAC

Pro analogový výstup na základě digitální hodnoty

__PWM__ - signál je zapnutý jenom několik % z periody

__Paralelní převodníky__ - několik odporů s vždy jednou tolik vyšším odporem než předchozí

#### RTC

Obsahují záložní zdroj

### Kontrolní otázky

Jaká je obvyklá organizace paměti?

> pracovní registry, univerzální registry, RWM data, program counter...

Jaké zdroje hodinového signálu se používají?

> krystal, RC obvod, keramický rezonátor

Jak probíhá reset?

> celý mikroprocesor na nastaví do předem definovaného stavu

Jakými způsoby se řeší ochrana proti rušení?

> galvanické stínění (a watchdog)

Jaké jsou vlastnosti IO bran?

> idk

Popište obecný princip fungování sériových rozhraní? jaké znáte?

> RS232 a RS485. Obě stany musí drzět stejný baud rate. V idle stavu je TX 1 a když se nastaví na 0, tak druhá strana musí začít číst z RX

K čemu slouží časovače a čítače a jak fungují?

> časovač je čítač, který reaguje na vnitří hodiny, čítač je jenom nějaký registr ve kterém se zvýší hodnota a při přetečení vyšle interrupt. dá se kombinovat s prescaler

Konstrukce a fungování základních ADC

> Komparační - rozdělí signál podle poměrů a měří hodnoty

Konstrukce a fungování základních DAC

> viz pár řádků nahoru

Speciální periferie

> Regulátor dobíjení baterie, IR přijimač a vysílač, LCD a LED obrazovky...

---

# Paměti PC

### Dělení

Podle typu přístupu: RAM, SAM (Serial access memory), stack, queue...

Podle možnosti zápisu: RWM, ROM, NVRAM, WOM, WORM (write once read many)

Podle typu elementární buňky: SRAM, DRAM, PROM, EPROM, EEPROM, FLASH

Podle uchování informace: Volatile (ztratí po ztátě napájení - XRAM), non-volatile (neztratí XROM)

### Dynamické paměti

Uložena v kondenzátoru - musí se pořád obnovovat (cca každých 10ms)

Paměťové buňky jsou ve čtvercové matici a několika vrstvách - dva dekodéry (row, column) pro výběr buňky

Adresy se dělí na dvě části: adresa řádku a adresa sloupce

Paměťový kontrolér řeší 3 věci:

1. rozdělení adresy z CPU na adresu řádku a sloupce
2. správné nastavení RAS (Row address strobe), CAS (Column address strobe), WE (Write enable) a READ signály
3. Přesouvá uložená data přijimá data k zapsání do paměti

> RAS - Kontrolní signál, který určuje, že se jedná o adresu řádku, CAS - viz RAS ale sloupec

#### Čtení a zápis

Každá buňka obsahuje kondenzátor a přístupový tranzistor, který slouží pro výběr(aktivaci) tranzistoru.

Paměťový kontrolér nejdříve aktivuje řádek, celý řádek se poté stane aktivovaný. Poté se vybere sloupec, který aktivuje všechny sloupce. Výsledek zkřížení těchto dvou aktivovaných vodiču je 8 jednotlivých paměťových tranzistorů.

__Čtení__ - sloupec se nastaví na poloviční napětí kondenzátoru a otevřou se tranzistory sloupce. To zapříčíní odchod napětí z kondenzátoru a napětí sloupce se zvýší. Tuto změnu poté zesílí zesilovač (zesílí napětí na napětí kondenzátoru - tzn automaticky se také provede refresh). IO buffer poté přečte tyto napětí a pošle to zpět do CPU.

__Zápis__ - velmi podobný princip jako u čtení, s tím rozdílem, že se napětí zvýší ze strany zápisového bufferu

#### Další operační módy

Pro snížení přístupnosti se může v rámci "jedné" operace aktivovat několik sloupců po sobě.

Nejznáměnší mód je "Page mode." V tomto módu zůstavá adresa řádku stejná a mění se pouze číslo sloupce (rychlejší přístup protože se nemusí dekódovat adresa řádku). Čtení a zápis poté probíhá normálně.

#### Vylepšené typy DRAM

__SDRAM__ - Synchronní DRAM. Nižší přístupový čas a pracují v burst módu.

__DDR SDRAM__ - Double data rate SDRAM. Dvojnásobek přenosové rychlosti (dva kanály - vložení stejných RAM sticks do specifických slotů na MB). Zpětně kompatibilní s DRAM.

#### Refresh

Obnovení probíhá zhruba každých 10ms a při každém čtení. Tři refresh metody:

__RAS-only__ - Nejjednodušší a nejpoužívanější. Jakoby přečte celý obsah pamětí a tím se obnoví. Nutný externí obvod (nebo CPU samotné), který toto simuluje.

__CAS-before-RAS__ - Uvnitř DRAM čipu. CAS je udržován na nízké úrovní dokud RAS taky neklesne na nízkou úroveň a tím pádem se vykoná automatické obnovení.

__Hidden__ - Refesh je skrytý za normálním přistupem pro čtení. CAS je na nízké úrovni a mění se pouze RAS.

#### Paměťové moduly

__DIP__ (Dual inline package) - integrované obvody s vývody na obou stranách. Kvůli velikosti se používají spíš DIMM a SIMM.

__SIMM__ - Single inline memory module. Mohou mít DIP až na obou stranách a to s 30 nebo 72 piny. Dnes spíš 72 pinů -> 32 bitů

__DIMM__ - Dual in. mem. mod. - 168 pinů -> 64 bitů. Používají se v SDRAM a DDR-RAM

__RIMM__ - Rambus in. mem. mod. - 184 kontaktů, 64/128/256 MB, max 40MHz. Až 800MHz v případě DDR-RAM. Používané v Intel P4 rok cca 2000

### Statické paměti

Používaný klopný obvod místo kondenzátoru. 4 nebo 6 tranzistorů na paměťovou buňku.

Organizovány do 2D mřížky. Řádek = word

Adresa řárů jako jedna informace, není zde adresní multiplexing. Je za potřebí více pinů, tím pádem jsou čipy SRAM větší, ale o hodně rychlejší (jednodušší dekódovaní adresy). Není potřebný refresh. Jsou o dost dražší.

#### Složení buňky

Dva NMOS přístupové tranzistory a dva NMOS paměťové. Poté tam jsou dva odpory nebo dva PMOS tranzistory, které tvoří s paměťovými tranzistory CMOS dvojici

#### Čtení a zápis

Není zde adresní multiplexing a dekodér aktivuje odpovídající adresové vodiče. Při čtení se aktivují přistupové tranzistory a tím propojí paměťové tranzistory s datovou linkou. Po stabilizaci vybere dekodér odpovídající sloupec a předá data do IO bufferu, který to poté předá do CPU.

Zápis dat probíha naopak. Přes vstupní datový buffer se data zesílí čtecím zesilovačem a zároveň se aktivuje dekodér řádku a přístupové tranzsitory. Čtecí zesilovač je silnější než paměťový tranzistor a tím pádem se data zapíšou.

#### Async, Sync a PB SRAM

__Async SRAM__ - L2 cache, není synchronizována systémovými hodinami (proto async), a proto musí CPU na data čekat (ne dlouho)

__Sync burst SRAM__ - Synchronizována se system clock

__Pipeline burst SRAM__ - Požadavky na SRAM se zřetězí (shromáždí) tak, aby se zapsaly téměř okamžitě. Je navržena pro práci se sběrnicí

### ROM

Buňka je představována jako odpor nebo pojistka. Výrobce při naprogramovaní některé z nich elektronicky přepálí a tím pádem někeré prvky vedou proud (logická 1) a některé ne (logická 0). Data si pamatuje napořád

### PROM

Data jsou zapisována elektronicky pomocí programátoru (ne člověk, ale zařízení). Potom podobně jako u ROM se přepálí pojisty a nelze do nich znovu zapisovat.

### EPROM

Lze do nich opakovaně zapisovat. Uchovává informace pomocí elektrického náboje, který je izolovaný a dokáže ho udržet. Maže se pomocí UV světla (v pouzdře je okénko kudy se UV světlo dostane). Doba pamatování je omezena na 10 až 20 let.

### EEPROM

Velmi podobné EPROM, ale maže se elektronicky. Opět 10 až 20 let.

> U EPROM a EEPROM se vždy maže všechno. Pulz mazání a programování je zhruba 50ms

### Flash

Vychází z EEPROM, ale vyžaduje jenom 10us pulz pro mazání a programování. 10k a více mazacích cyklů, 10-100 let doba uchování.

Tím, že vychází z EEPROM, tak se také musí mazat všechno. Flash paměť potřebuje elektroniku navíc (adresní dekodér a buffer). Flash má malé bloky EEPROM paměti a při zápisu "1" se zkopíruje celý blok do bufferu/RAM, smaže se a data se z bufferu/RAM zkopírují zpět na původní místo.

### Další typy pamětí

#### VRAM

DRAM s šiřším přenosovým pásmem + dva porty: jeden na obnovování obrazu, druhý pro zápis -> tj. zdvojnásobení pásma

#### WRAM

Pobodná VRAM (dvouportový DRAM), ale má ještě vyšší přenosové pásmo a několik grafických funkcí (např double-buffering)

#### SGRAM

Sychnronní grafická RAM - optimalizovaná pro nejvyšší možný přenos dat

#### FIFO

Většinou přímo v MCU. Dva typy:

* bez přesouvání obsahu - zápis a čtení pomocí dvou registrů (čte se ze začátku a zapisuje se na konec) + dva registry na indikaci prázdné a plné fronty
* s přesouváním obsahu - přídavný registr a při jakékoliv operaci se posune zbytek fronty o jedno nahoru nebo dolů

#### Cache

L1 - přímo v CPU - zásobuje daty se sběrnice - dělá overfetching
L2 - mezi CPU a RAM - všechna data jsou tudy, obsahuje řadič, který předpovídá co dalšího by mohl procesor chtít

> Tři režimy cache:
>
> 1. Write-through - data se zapisují do cache a do paměti zároveň, při čtení se podívá jestli nemá data v cache. Nejpomaljší a nejstarší způsob
> 2. Write-back - data se zapisují pouze do cache a až po nějaké době se zapisují do ram. než se zapíšou do ram, můžou se několikrát změnit
> 3. Pipeline burst - nejnovější a nejrychlejší. pokud čte, přečte i adresy za požadovanou (protože by to stejně dělal za chvíli).

### Chyby pamětí, detekce a oprava

Občas se stane, že vrátí nesprávná data, např DRAM je méně spolehlivá než SRAM. Dva druhy chyb:

* opakující se/trvdé chyby - HW chyba, jsou konzistentní a jednoduché na identifikaci
* přechodné/měkké chyby - stávají se náhodně a jsou častější než tvrdé chyby. hůře se identifikují

#### Parita

Fyzická parita - paritní bity jsou předány paměťovému kontroleru a jsou uložené s daty.

Logická parita - při čtení se paritní bity generují a nemůže zde nastat chyba u paritního bitu.

Paměťové moduly jsou buď s paritou nebo bez.

Moduly bez parity obsahují jeden bit paměti a jeden bit uložených dat.

Moduly s paritou mají na každých 8 bitů jeden bit navíc.

__Kontrola parity__

Obvod spočítá počet nul a jedniček v bytu a pokud je sudý počet jedniček, uloží do devátého bytu 1, pokud lichý tak 0. To zapříčíní vždy lichý počet jedniček a při čtení to lze jednoduše zkontrolovat. Tento způsob umožňuje kontrolovat chybu pouze jednoho bitu, pokud se prohodí dva, zůstává počet jedniček stále stejný.

#### ECC

(Error correction code)

Pozná chyby ve více bitech a umí opravit jeden. ECC používá skupiny bitů: 7 na 32 a 8 na 64. Tím může detekovat chyby až 4 bitů (bývají velmi vzácné tak velké chyby)

ECC snižuje výkon systému cca o 2-3%

---

# Externí paměti

### Magnetické

Médium má tvar kruhové desky nebo dlouhé pásky a je pokryta magentickou vrstvou. Pohybuje se konkstatní rychlostí.

---

#### Pevný disk

Obsahuje plotny diskového tvari - slitina hliníku nebo skla.

__Části disku__ - plotny, hlavy pro RW, pohon hlav, vzduchové filtry, pohon ploten, řídící deska, kabely a konektory

__Geometrie disku__

Data jsou ukládána v bytech - ty jsou uložené po sektorech o velikostech 512 bytů (nejmenší zapisovatelná jednotka) - sektory jsou ve stopách - stopy jsou v cylidrech

__Stopy__ - každá strana plotny ja rozdělena na soustředné stopy (kružnice). počet se neustále zvyšuje

__Cylindy__ - stopy nad sebou

__Sektory__ - stopa rozdělená na několik výcečí

#### Disketové mechaniky

Rychlost 300 rpm.

__Části__ - hlavy pro RW, pohon hlav a mechaniky, řídící deska, řadič, konektory

---

#### Optické paměti

__Ukládání dat__

Záznam je v podobě prohlubní a ostrůvků (pitů a polí) a jedičky se zapisují jako přechod mezi pitem a polem - datové nuly se nezapisují.

Čtení - prochází médium 2x a pokud se na místě nachátí pit, dojde k částečnému rozptýlení odrazu a to změní odezvu do čtecího senzoru

__CD-ROM__

Při IO nedochází ke kontaktu s médiem - laser a fotodetektor. Měří se intenzita odráženého světla. Mechanismus laseru je od disku asi 1mm. Stopa je spirála od středu a přehrávač musí měnit rychlost otáčení disku aby zajistil konstantní rychlost. Začátek čtení je od vnitřních stop k vnějším. Mechanika se prvně otáčí 540 rpm a na okraji už jenom 212 rpm - tím se vyrobí konstantní otáčení.

__Části__ - laserová hlava, fotodetektor, servomechanismus, ovládácí čip, vyrovnávací paměť, elektronika pro dekédování signálu a řízení procesů

__DVD__

"Vysokokapacitní CD". Používá se laser s vlnou 650nm (CD 780nm).

---

#### Magneto-optické paměti

__Ukládání dat__

Kombinace magnetického a optického disku (nečekaně), často se používá označení *termomagnetooptický disk*. Využívá lokální změny magnetické orientace, která vzniká za působení tepla a elektromagnetického pole současně.

dvě velikosti - 3.5" (1.3 GB) a 5.25" (5.2 GB)

Uchovává data až 30 let

__Geometrie disku__

Podobně jak u CD - spirála a otáčí se konstatní rychlostí. Stopa je rozdělena na 25 sektorů po 512 nebo 1024B.

---

# Zobrazovací technologie

## CRT

vakuová skleněná baňka, jejíž přední část tvoří stínítko potažené luminoforem.

Proces zobrazování začíná u elektronového děla, které je koncem každé katodové trubice. Jednotlivé elektronové svazky jsou emitovány z nepřímo žhavené katody, která má na svém povrchu nanesenou emisní vrstvu. Po zahřátí se vystřelí proud elektronů. Před dopadem na obrazovku také projdou filtrem, který propustí jen někeré a to reguluje intenzitu. Na horním okraju trubice je anoda s vysokým napětím, která vytahuje elektrony z děla. Dále tam je vyhylovací cívka, která vychyluje elektrony do správněho směru.

Paprsek elektronů začíná v levém horním roku a postupně dojde na pravý horní roh kde se na chvíli vypne, přesune o řádek níž doleva. Tento proces se nazývá *rastrování* nebo *řádkování*. Takhle projede paprsek celou obrazovku až do dolního pravého rohu, kde se vypne a přesune do levého horního rohu. Refresh probíhá asi 60x za sekundu.

### Invar

Otvory v macse jsou kruhové a uspořádany do trojúhelníků. Nevýhoda je velká plocha, která je tvořena kovem masky - větší náchylnost k tepelné roztažosti.

Invarová maska je část kulové výseče. I přes všechny vylepšení, invarové masky mají toto klenutí stále nepřehlédnutelné.

### Trinitron

Kovový plát masky je zaměněn za konstrukci pevně natažených drátků umistěných v horizontálním směru obrazovky. Díky tomu tudy proníká více elektronů, tudíž je jasnější obraz.

Dále je potřeba použít dva vetrikální drátky v třetině obrazovky, které udržují mřížku na místě.

Obrazovka je spíč šířší než vyšší a nedochází k vertikálnímu zkreslení, proto není nutné zakřivovat obrazovku na výšku.

### Výhody a nevýhody

__Výhody__ - Ostrost, věrohodné barvy, nízká odezva, pozorovací úhly

__Nevýhody__ - Velikost, spotřeba, vyzařování

---

## LCD

Základ jsou tekuté krystaly.

Jádrem je TN (twisted nematic) struktura, která je z obou stran obklopena polarizačními vtrstvami.

Nepolarizované světlo projdfe prvním pol. sklem a polarizuje se. Poté prochází vrstvami tekutých krystalů, které světlo otočí o 90° a nakonec projde druhým sklem.

Takto se chová v klidovém stavu - propouští světlo. Jakmile začne krystaly protékat proud, krystalická struktura se začne orientovat podle směru toku proudu a druhá polarizační vrstva začne světlo blokovat.

Vrstva tekutých krystalů je rozdělena na malé buňky stejné velikosti a musí být podsvícená bílým světlem (nejčastěji se používají elektroluminescenční výbojky, které začínají být nahrazovány bílými LED)

### Barevné LCD

Velmi podobná jednobarevných, ale každý bod se skládá ze tří menších. Tyto body obsahují červený, zelený a modrý filtr umístěné vedle sebe na horní skleněné destičce. Barva vzniká tím, že z bílého světla propouští jenom některé složky různou intenzitou

### Pasivní displeje

Pasivní matice má mřížku vodičů s body nacházející se na každém průsečíku v mřížce. Proud protéká dvěmi vodiči v mřížce a tím aktivuje každý bod. Impuls projede každým řádkem a když je sloupec uzemněný, vznikne elektrické pole, které změní stav kapalného krystalu (z bílého na černý).

Problém je při nárustu počtu řádků a sloupců, protože velikost elektrody musí být menší a napětí větší. Toto se projevuje neostrým obrazem, nižším kontrastem a vysokou odezvou (150-250ms) (viditelná stopa kurzoru, nebo úplným vymizením při rychlém pohybu)

### Aktivní displeje

Známe jako TFT (Thin Film Transistor)

Matice má na každém průsečíku tranzistor nebo diodu, takže potřebují méně proudu na aktivaci. Tím se zvýší refresh rate a také přesnost ovládání svítivosti každého bodu. TF tranzistory kompleteně izolují jeden bod od ostatních.

### Výhody a nevýhody

__Výhody__ - Kvalita obrazu, životnost, spotřeba, odrazivost a osilnivost, bez emisí

__Nevýhody__ - Citlivost na teplotu, pevné rozlišení, vadné pixely, doba odezvy

## Plazmové monitory

### Princip

V klidovém stavu se v plazma displejích nachází směs plynu, které vyzařují jednu barevnou složku. Jelikož to jsou __neutrální__ atomy, musí se z nich nějak vytvořit plazma.

Ta se vytváří pomocí zavedení proudu do plynu, čímž se objeví volné elektrony. Srážkou mezi elektronem a plynem se ztratí elektrony u některých atomů a vzniknou kladně nabité ionty. Spolu s elektrony získáváme plazmu.

Jednotlivé navíté částice se začnou pohybovat k opačným pólům. V plazmě dochátí k velkým pohybům a ve vzniklém "zmatku" začnou částice narážet do sebe.

Při nárazu získají částice iontu energii a hned poté ho elektromagnetické síly donutí k návratu zpět a přebytečná energie se uvolní ve formě fotonu (světla).

Celý displej je tvořen maticí fluerescentních buněk ovládané sítí elektrod. Horizontální řádky tvoří adresovací elektrody a vertikální sloupce zobrazovací elektrody.

Buňky jsou uzavřeny mezi dvěma tenkými sklěmenými tabulkami a každá obsahuje malý kondenzátor a tři elektrody. Adresovací elektroda je na zadní stěně a dvě transparentní zobrazovací elektrody na přední.

Do obou zobrazovacích elektrod je puštěno střídavé napětí a to začne ionizovat plyn.

Ovládání intenzity je pomocí PWM.

### Výhody a nevýhody

__Výhody__ - kvalitní a kontrastní obraz, bez podsvícení, velké pozorovací úhly, minimální hloubka a ostrost

__Nevýhody__ - Paměťový efekt, levnější displeje mají problém s kontrastem, cena, vyšší spotřeba

## OLED

"Organic Light Emmiting Diode"/"Organic Light Emmiting Display"

### Princip

Základem je kovová katoda na které je několik vrstev:

1. Vodivá vrstva pro elektrony
2. Organická vrstva emitující světlo
3. Vrstva pro přenos děr
4. Průhledná anoda
5. Ochranná sklěněná vrstva

Po přivedení napětí na obě elektrody se začnou elektrony hromadit na straně organické vrstvy k anodě. Díky (představující kladné částice) se hromadí na opačné straně blíže k anodě.

V organické vrstvě se elektrony začnou srážet a dojde k jejich vzájemné eliminaci, která vypustí energii ve formě fotonu. Tento jev se nazývá __rekombinace__.

### AMOLED a PMOLED

Active/Passive Matrix OLED

Stejný princip jako u aktivního a pasivního LCD. Doby jsou organizovány do pravoúhlé matice.

U pasivních displejů je každá oled aktivována dvěmi na sebe kolmými elektrodami, které prochází celým displejem.

U aktivních je oled aktivována vlastním tranzistorem

### Vlastnosti, výhody, nevýhody

Nepotřebují podsvětlení, méně energeticky náročné

__Výhody__ - Vysoký kontrast, tenké, plně barevné, nízká spotřeba, dobré pozorovací úhly, nulové zpoždění, snadná výroba, možnost instalace na pružný podklad

__Nevýhody__ - cena, v dnešní době pouze u malých displejů

## Eink

### Princip

Jednotlivé body jsou uzavřené malé kapsle o velikosti desítek či stovek um. Kapsle obsahují elektroforetický (elektricky separovatelný) roztok. V tomto roztoku jsou černé částice nabité záporně a bíle částice nabité kladne.

Kapsle jsou umístěny mezi elektrody a přivedením napětí na horní a dolní oblast se částice náboje přitáhnou k opačnému pólu.

Získáváme tak zvrchu bíle body. Lze také vytvořit několik odstínů šedé.

Elektroforetický roztok musí být dlouhodobě chemicky stabilní a průhledný roztok používá hydrokarbonový olej. Černé částice jsou vyrobeny z uhlíku.

Pro změnu rozmístění částic je potřeba velmi malý proud (několik desítek nA) a napětí 5-15 V. Po odpojení napájení částice zůstavají na svém místě.

Pro zobrazení barev se před kapsli umístí jeden barevný filtr.

### Výhody a nevýhody

__Výhody__ - dobrý kontrast, čitelnost na přímém slunci, není nutné podsvětlení, skoro 180°pozorovací úhel, velmi tenké, možno používat na pružném podkladu, nulová spotřeba po vykreslení a minimální při překreslení

__Nevýhody__ - 16 odstínů na kanál (16 barev při černobílém a 4096 při barevném) -> špatné barevné podání, dlouhé překreslení (stovky ms)

# CUDA

Masivní paralelismus

__Výhody__ - několik stovek až tisíců vláken, nezávislé vlákna, dobré na výpočty, optimalizované na sekvenční přístup k datům

__Nevýhody__ - "if" bloky zpomalují, určené pouze k výpočtům

Větší počet ALU na jeden instrukční dekodér, mají menší taktovací frekvenci oproti CPU a propustnost sběrnice je násobně vyšší

## Architektura

Katra je rozdělena na __Multiprocesory__. Paměť karty je jenom jedna pro všechny procesory a každý procesor má své registry. Každý multiprocesor má svoji cache pro zrychlení přistupnosti k texturám

Paměť obsahuje tři části: Hlavní paměť, texturová paměť a paměť na konstanty

## Obecné

CPU paměť - device memory

GPU paměť - host memory

Tyto paměti nejsou sdílené a pro přístup se musí vždy obsah kopírovat z jedné do druhé

Unified memory - Automaticky kopíruje data

Pro zvýšení výkonu:

* kopírovat data jenom když je potřeba - ideálně jenom 2x
* používat GPU jenom k výpočtu
* pro přesuny mezi CPU a GPU použít pipelining
* používat sekvenční přístup
* ideálně aby každé jádro vykonávalo stejné instrukce (bez ifů)
* vybrat optimální velikosti mřížek jader

> CUDA je pouze framework. Existují bindings pro C/C++, Python, Java...

CUDA má vlastní extension pro LLVM IR a vlastní kompilátor *nvcc*. Většina datových typů je stejných, ale jsou nějaké přidané, např:

* int3 - obsahuje x,y,z
* int3 - x,y,z,w

CUDA dává k dispozici globální konstanty:

* `dim3 gridDim` - rozměry gridu
* `uint3 blockIdx` - pozice v gridu
* `dim3 blockDim` - velikost bloku
* `uint3 threadIdx` - pozice v bloku
* `int warpSize` - warp size ve vlákne