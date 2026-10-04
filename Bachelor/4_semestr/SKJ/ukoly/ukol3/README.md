<div>
<p>Stáhněte si <a href="/task/SKJ/2024S/GAU01/zadani_cv03/asset/template.zip">template</a> se zdrojovými kódy cvičení.</p>
<p>Pracovat budete se souborem <code>atoms.py</code>. Soubor <code>playground.py</code> zajišťuje vytvořeni GUI a timeru.
Několikrát za sekundu je volána metoda <code>tick</code>, která vykreslovacímu jádru předává souřadnice atomů, které jsou pak vykresleny do plátna.
Naleznete ji ve třídě <code>ExampleWorld</code> v souboru <code>atoms.py</code>.</p>
<p><strong>Úkol 1 (2 body):</strong></p>
<p>Doplňte inicializátor třídy <code>Atom(pos, vel, rad, color)</code>.
Doplňte funkci <code>to_tuple</code>, aby vracela hodnoty atomu v <code>tuplu</code>.</p>
<p>V konstruktoru třídy <code>ExampleWorld</code> jsou předány parametry <code>no_atoms</code> a <code>no_falldown_atoms</code>,
které reprezentují počet dvou druhů atomů, které se mají vytvořit,
k tomu šířka a výška plátna (<code>size_x</code>, <code>size_y</code>). V metodě <code>random_atom</code> vygenerujte a vraťte atom s náhodným poloměrem a pozicí a zelenou (<code>green</code>) barvou 
(při pozici vemte v úvahu šířku a výšku plátna, ať se nevykresluje mimo).
V konstruktoru třídy <code>ExampleWorld</code> použijte a naimplementujte funkci <code>generate_atoms</code>,
která vytvoří odpovídající počet náhodných atomů, které uloží do seznamu.
Pro generování můžete využít funkci <code>random.randint</code>.</p>
<p>Rozpohybujte atomy, a to tak, že k souřadnici <em>x</em>/<em>y</em> budete přičítat číselnou hodnotu (rychlost pro směr <em>x</em>/<em>y</em>).
Rychlosti předejte v konstruktoru třídy <code>Atom</code> (dohromady tedy bude mít konstruktor parametry <code>pos</code> (tuple dvou hodnot), <code>vel</code> (tuple dvou hodnot), <code>rad</code>, <code>color</code>).
Ve třídě <code>Atom</code> naimplementujte metodu <code>apply_speed</code> která posune daný atom o vektor rychlosti - tuto metodu je pak potřeba volat v metodě <code>tick</code>.
Při každém zavolání metody je tedy atom posunut a převeden na n-tici (tuple), což způsobí, že se atom bude po plátně pohybovat.
V metodě <code>apply_speed</code> kontrolujte, zda se atom dostal na(za) hranici plátna.
V okamžiku, kdy se tak stane, otočte směr jeho pohybu - otočte vektor rychlosti ve směru <em>x</em> nebo <em>y</em>,
podle toho, na kterou hranu atom narazil (velikost plátna předejte do funkce v argumentech).</p>
<p><strong>Úkol 2 (2 body):</strong></p>
<p>Vytvořte třídu <code>FallDownAtom</code> dědící z třídy <code>Atom</code> (pokud dědění nepoužijete, není to problém).
Tato třída bude obsahovat dvě třídní proměnné <code>g</code> a <code>damping</code> (například s hodnotami <code>3.0</code> a <code>0.7</code>), které představují gravitační zrychlení a odrazový útlum.
Třída <code>FallDownAtom</code> dědí metodu <code>apply_speed</code>. Tu je však potřeba změnit tak, aby na tento atom působila gravitace.
Při každém zavolání této metody se provede přičtení hodnoty <code>g</code> k vertikální rychlosti
(jelikož je počátek osy <em>y</em> nahoře, přičítáním hodnoty <code>g</code> k rychlosti atomu bude atom směrem dolů zrychlovat, přičítáním k záporné rychlosti <em>y</em>,
tedy při stoupání atomu vzhůru, zase bude atom zpomalovat, až se dostane rychlost do kladných hodnot a začne opět klesat).
Atom navíc při nárazu na spodní hranici zpomalí ve vertikální i horizontální rychlosti (<code>rychlost</code> * <code>damping</code>), to způsobí,
že se atomy nebudou pohybovat nekonečně dlouho po plátně.
Naimplementujte metodu <code>random_falldown_atom</code> v třídě <code>ExampleWorld</code> tak, aby generovala náhodny atom, na který působí gravitace.
Také tuto metodu použijte v metodě <code>generate_atoms</code>, aby přidala do seznamu atomů atomy, na které působí gravitace.</p>
<p><strong>Úkol 3 (1 bod):</strong></p>
<p>Implementujte metody <code>add_atom</code> a <code>add_falldown_atom</code>, která vytvoří nový padající atom.
Tyto metory jsou volány při události kliknutí levým nebo pravým tlačítkem myši.</p>
</div>