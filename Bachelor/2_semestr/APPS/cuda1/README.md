# **Zadní**

Seznámení s technologií CUDA, příprava způsobu práce pro další týdny (každý dle svých možností).

Dle vzoru příkladů z github si vždy připravte kernel a funkci, ze které se bude kernel spouštět. Všechny následující příklady implementujte do dvou souborů: jeden .cu a druhý .cpp.

1. Dle příkladu cuda2_unm si naimplementujte funkci a kernel, který provede sečtení dvou vektorů prvků float a výsledek se vrátí ve třetím vektoru. Délka vektorů minimálně 1000000 prvků. Zvolte si prvky vektorů tak, aby jste následně dokázali snadno v programu zkontrolovat, že výsledek je správně. Ne ale primitivní zadání typu: všechny prvky pole jsou 0 nebo 1, nebo všechny stejné.

2. Napište si funkci a kernel, který převede řetězec na velká písmena. Načtěte si svůj zdrojový kód *.cpp,  převeďte ho na velká písmena a uložte jako *.up. Pro převod na velká písmena si pro kernel připravte vhodné pole znaků (konverzní tabulku), aby kernel neobsahoval pro samotný převod žádný if.

3. Napište si funkci a kernel pro násobení dvou matic a výsledek se uloží do třetí matice. Rozměr matic minimálně 1000x1000 a typ prvků double. Matice bude mít pevně danou velikost a bude tak tvořit jeden souvislý blok dat. Druhou matici si před násobení transponujte, aby násobení neprobíhalo standardně řádek x sloupec, ale řádek x řádek. Kernel bude obsahovat jedinou smyčku for pro výpočet jednoho prvku matice (všechny kernely tak budou vykonávat stejný kód). Prvky v obou maticích si připravte tak, aby se následně dalo snadno (automaticky) zkontrolovat, že výsledek je správně. (Ne např. všechny prvky matice 0 nebo 1 a ve výsledku všechny prvky stejné hodnoty).