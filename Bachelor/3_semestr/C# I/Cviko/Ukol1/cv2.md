### Cvičení 2

Cílem cvičení bude implementovat padající kostičky tetrisu a to pomocí 4 různých variant reprezentace těchto kostiček/scény.

1. Založte nový projekt typu konzolová aplikace ve vámi zvoleném IDE.
2. Implementace pomocí vícerozměrného pole.
	1. Vytvořte třídu **MultidimensionalArray** a v ní metodu **Run**, která bude sloužit jako vstupní bod pro spuštění dané implementace. V rámci metody **Run** alokujte vícerozměrné pole logických hodnot, které bude reprezentovat hrací plochu (true = kostička, false = prázdné místo).
	2. Implementujte dvě metody pro vytvoření dvou různých typů kostiček (například *T* a *Z*). Tyto metody budou přijímat jako parametr X, Y souřadnice budoucí kostičky a pole ve kterém se má dané kostička vytvořit. Metody použijte v metodě **Run**.
	3. Vyzkoušejte si přidání breakpointu a zkontrolujte, zdali se vám kostičky v poli skutečně vytvořily.
	4. Implementujte metodu **MainLoop**, která bude přijímat jako parametr naše dvojrozměrné pole. V rámci metody implementujte kód, který na základě předaného pole provede vykreslení scény do konzole. Jednotlivé kostičky vykreslete pomocí Unicode znaku **\u2588** (jedná se o obdélník - čtverec získáte vykreslením 2 těchto znaků vedle sebe). Metodu **MainLoop** následně zavolejte z metody **Run** (ověřte, že vykreslení funguje). Při implementaci využijte metody a vlastnosti *Console.Clear*, *Console.SetCursorPosition*, *Console.ForegroundColor*, *Console.BackgroundColor* a *Console.CursorVisible*.
	5. V rámci metody **Run** spusťte metodu **MainLoop** v nekonečné smyčce. Po každém zavolání metody **MainLoop** spusťte tento kód: *Thread.Sleep(1000);*
	6. V rámci metody **MainLoop** implementujte posun kostiček (při každém zavolání metody dojde k posunu o 1 směrem dolů).
3. Implementace pomocí zanořených polí.
	1. Vytvořte třídu **JaggedArray**, která bude reprezentovat danou implementaci.
	2. Implementujte identickou logiku jako v bodě 1., ale tentokrát s využitím zanořených polí.
4. Implementace pomocí pole struktur.
	1. Vytvořte třídu **StructureArray**, která bude reprezentovat danou implementaci.
	2. Vytvořte strukturu **Brick**, která bude reprezentovat jednu kostičku (jeden čtvereček) a bude mít svou X a Y souřadnici v rámci složené kostičky (viz. další bod).
	3. Vytvořte strukturu **CompositeBrick**, která bude reprezentovat složenou kostičky (konkrétní tvar). Tato struktura bude mít svou X a Y souřadnici v rámci scény a bude obsahovat pole kostiček (struktura **Brick**).
	4. Implementujte identickou logiku jako v bodě 1., ale tentokrát s využitím pole výše zmíněných struktur. Vaše implementace musí splňovat následující:
		- metody pro vytváření složených kostiček budou přijímat jako parametry X a Y souřadnice a odkaz do pole složených kostiček, kde se má daná složená kostička vytvořit.
		- pro posun složených kostiček vytvořte samostatnou metodu, které předáte složenou kostičku a ona provede její posun.
5. Implementace pomocí pole tříd.
	1. Vytvořte třídu **ClassArray**, která bude reprezentovat danou implementaci.
	2. Implementujte identickou logiku jako v bodě 4., ale tentokrát s využitím tříd.
	