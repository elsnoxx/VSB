

### Cvičení 5
Cílem cvičení bude implementovat generátor map na základě dat z OSM.

1. Vytvořte konzolovou aplikaci s názvem **OSMMap** a knihovnu tříd s názvem **OSMMapLib**.
2. V knihovně **OSMMapLib** vytvořte třídy **Tile**, **Layer** a **Map**. 
3. Třída **Tile**:
	1. Bude mít vlastnosti (property) **X**, **Y**, **Zoom** a **Url**. U vlastnosti **Zoom** zajistěte aby ji nebylo možné nastavit na hodnotu menší nežli 1 (v případě přiřazení menší hodnoty dojde k nastavení na hodnotu 1).
	2. Všechny uvedené vlastnosti bude možné měnit pouze zevnitř třídy **Tile**.
	3. Bude mít konstruktor přijímající hodnoty pro **X**, **Y**, **Zoom** a **Url**.
	4. Bude přetěžovat metodu **ToString**. Uvnitř této metody dojde k naformátování popisu dlaždice v tomto formátu: "*[X, Y, Z]: url*". Při implementaci využijte **StringBuilder**.
4.  Třída **Layer**:
	1. Bude mít vlastnosti **UrlTemplate** (string) a **MaxZoom** (int). Tyto vlastnosti bude možné nastavit pouze z vnitřku třídy. Číst je půjde odkudkoliv.
	2. Bude obsahovat konstruktor s možností nastavení urlTemplate a maxZoom. Obě tyto vlastnosti budou nepovinné a budou mít výchozí hodnoty *"https://{c}.tile.openstreetmap.org/{z}/{x}/{y}.png"* (šablona URL) a 10 (maximální zoom).
	3. Bude obsahovat metodu **FormatUrl**, která bude přijímat parametry **x**, **y** a **zoom** (vše int). Tato metoda bude vracet URL konkrétní dlaždice mapy. URL bude vygenerována nahrazením složených závorek z šablony URL za konkrétní hodnoty. Hodnotou pro *{c}* bude náhodně vygenerované písmeno *a*, *b*, nebo *c*.
	4. Bude obsahovat indexer přijímající hodnoty **x**, **y** a **zoom** (int) a vracející novou dlaždici (Tile) na daných souřadnicích x a y.
5. Třída **Map**:
	1. Bude obsahovat veřejnou vlastnost **Layer**.
	2. Bude obsahovat vlastnosti **Lat** (double), **Lon** (double) a **Zoom** (int). Při nastavení vlastnosti **Lat** normalizujte její hodnotu na rozmezí -90 až +90. Při nastavení vlastnosti **Lon** normalizujte její hodnotu na rozmezí -180 až 180. Při čtení vlastnosti **Zoom** bude vždy vrácena hodnota v rozmezí 1 až **MaxZoom** z aktuální vrstvy (Layer).
	3. Bude obsahovat neveřejnou vlastnost **CenterTileX** (int), která bude jen pro čtení a bude vracet výslede tohoto výpočtu: `(int)((lon + 180.0) / 360.0 * (1 << zoom))`
	4. Bude obsahovat neveřejnou vlastnost **CenterTileY** (int), která bude jen pro čtení a bude vracet výslede tohoto výpočtu: `(int)((1.0 - Math.Log(Math.Tan(lat * Math.PI / 180.0) + 1.0 / Math.Cos(lat * Math.PI / 180.0)) / Math.PI) / 2.0 * (1 << zoom))`
	5. V konzolové aplikaci referencujte knihovnu **[MapRendererLib.zip](https://vsb.sharepoint.com/:u:/r/sites/C12/Sdilene%20dokumenty/General/Practical%20Lessons/MapRendererLib.zip?csf=1&web=1&e=H57fb0)** (součást zadání). 
	6. Třída **Map** bude obsahovat metodu **Render** přijímající název souboru do kterého se bude mapa generovat. Tělo této metody bude vypadat následovně:
```
		MapRenderer mapRenderer = new MapRenderer(4, 4);
        for (int x = -2; x < 2; x++)
        {
            for (int y = -2; y < 2; y++)
            {
                Tile tile = this.Layer[this.CenterTileX + x, this.CenterTileY + y, this.Zoom];

                Console.WriteLine(tile);

                mapRenderer.Set(x + 2, y + 2, tile.Url);
            }
        }
        mapRenderer.Flush();
		mapRenderer.Render(fileName);
```		
6. Následně otestujte vaši implementaci. Pro otestování vytvořte novou mapu a přiřaďte ji vrstvu. Následně na mapě zavolejte metodu Render.
7. Zkuste vytvořit další vrstvu (šablona *"https://b.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png"* a zoom *17*). A vykreslit mapu s ní.

### Domácí úkol
1. Implementujte minimálně 2 různé vrstvy (každá bude používat jinou šablonu URL). Každá vrstva bude mít navíc vlastnosti **Opacity** (číslo v rozmezí 0.0 - 1.0), které bude udávat průhlednost vrstvy.
2. Upravte mapu tak aby umožňovala nastavení více vrstev současně.
3. Upravte metodu **Render** tak aby vykreslila všechny vrstvy. Metodu **Flush** je nutné volat po nastavení dlaždic každé vrstvy. Metodu **Render** je potřeba volat až po nastavení všech vrstev. Průhlednost dlaždic nastavíte jako čtvrtý parametr metody **Set**.
4. Nastavte mapě jednotlivé vrstvy (včetně průhlednosti).
5. Otestujte vykreslení mapy s vrstvami.