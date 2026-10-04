# **Zadání**

Seznámení s technologií CUDA, příprava způsobu práce pro další týdny (každý dle svých možností).

Dle vzoru příkladů z github si vždy připravte kernel a funkci, ze které se bude kernel spouštět. Všechny následující příklady implementujte ve dvou souborech: jeden .cu a druhý .cpp.

Pro další přípravu použijte příklady cuda3 či cuda4, dle vlastního uvážení.

1. Načtěte si barevný obrázek, ne čtverec, min 300x200 a rozložte si ho na 3 obrázky, kde každý bude obsahovat jen jednu barvu. Výsledné obrázky musí být také RGB, ale bude v nich vyplněna jen jedna barva.

Kernel bude mít 4 argumenty: vstupní obrázek a 3 výstupní obrázky. Všechny stejné velikosti.

2. Implementujte flip (mirror) obrázku, volitelně horizontálně či vertikálně. Pozor na velikost mřížky, ta musí být jen přes půl obrázku, aby se body neprohodily 2x.

Kernel bude mít jen 2 parametry: obrázek a volbu hor_vert.

3. Otočte obrázek o 90°¸ volitelně ve směru hodinových ručiček a proti směru. Obrázek nesmí být čtverec.

Kernel musí mít 3 parametry, protože dojde k prohození šířky a výšky: vstupní obrázek, výstupní obrázek a směr rotace.

Určitě není pro otočení o 90° nutno používat sin/cos.

4. Připravené funkce použijte následovně:

    1. Načtěte obrázek, proveďte jeho flip horizontálně i vertikálně (dojde k rotaci o 180°) a výsledek zobrazte.

    2. Otočte obrázek z kroku a) doleva o 90° a zobrazte. Otočte obrázek z kroku a) doprava o 90° a zobrazte.

    3. Obrázek otočený doprava rozložte na 3 barvy a zobrazte.