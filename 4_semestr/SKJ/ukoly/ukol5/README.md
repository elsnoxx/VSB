<div class="tab-pane active" id="tab_assignment">
                <div>
<p>Zkusíme si, jak pracovat s XML.</p>
<p>Dokumentaci a tutoriál knihovny <code>xml.etree.ElementTree</code> naleznete <a href="https://docs.python.org/3/library/xml.etree.elementtree.html">zde</a>.</p>
<h2>Materiály:</h2>
<p><a href="/task/SKJ/2023S/GAU01/c05/asset/template/tasks.py">tasks.py</a> - zadání<br>
<a href="/task/SKJ/2023S/GAU01/c05/asset/template/tests.py">tests.py</a> - testy<br>
<a href="/task/SKJ/2023S/GAU01/c05/asset/template/cheatsheet.py">cheatsheet.py</a> - tahák</p>
<p><a href="/task/SKJ/2023S/GAU01/c05/asset/assets/skj_class.xml">skj_class.xml</a> - xml soubor, se kterým se pracuje<br>
<a href="/task/SKJ/2023S/GAU01/c05/asset/assets/skj_class_formatted.xml">skj_class_formatted.xml</a> - čitelně formátovaný soubor<br>
<a href="/task/SKJ/2023S/GAU01/c05/asset/assets/skj_class_generator.py">skj_class_generator.py</a> - script k vygenerování xml souboru  </p>
<h2>Odevzdání</h2>
<p>Řešení úlohy odevzdávejte zde do Kelvinu (odevzdejte vyřešený soubor <code>tasks.py</code>).
Za každý správně naimplementovaný test dostanete 1 bod, tedy maximálně můžete získat 5 bodů.</p>
<!--
V p&#345;&#237;pad&#283;, &#382;e neodevzd&#225;te do konce hodiny,
m&#367;&#382;ete &#250;lohu dod&#283;lat doma a odevzdat do soboty 9:00 s 50% penalizac&#237;. Maxim&#225;ln&#283; m&#367;&#382;ete z&#237;skat 3 body za 6 spln&#283;n&#253;ch &#250;loh.

Podm&#237;nkou pro ud&#283;len&#237; bod&#367; za vypracov&#225;n&#237; doma je &#250;&#269;ast na hodin&#283;. Jen v p&#345;&#237;pad&#283;, &#382;e se v&#225;m to nepoda&#345;&#237; na hodin&#283;, tak m&#367;&#382;ete dod&#283;lat doma.-->

<h2>Další informace</h2>
<p>Základní informace o předmětu naleznete <a href="https://github.com/geordi/skj-course">zde</a>.</p>
<h3>Nastavení prostředí a testů</h3>
<p>Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).
Nainstalujte si do něj knihovnu <code>pytest</code>, abyste mohli spouštět připravené unit testy.
</p>
<pre class="highlight"><code class="language-bash hljs" data-highlighted="yes">$ python3 -m venv venv_dir_path     <span class="hljs-comment"># Vytvoří virtuální prostředí pro instalaci balíčků (spusťte pouze jednou)</span>
$ <span class="hljs-built_in">source</span> venv_dir_path/bin/activate <span class="hljs-comment"># Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)</span>
    <span class="hljs-comment"># na windows je cesta k activate scriptu: venv_dir_path/Scripts/Activate s příponou .ps1 nebo .bat dle konzole</span>
(venv) $ pip install pytest         <span class="hljs-comment"># Nainstaluje balíček pip do virtuálního prostředí (spusťte pouze jednou)</span>
(venv) $ python -m pytest tests.py  <span class="hljs-comment"># Spustí testy ze souboru tests.py</span>
(venv) $ python -m pytest -v tests.py   <span class="hljs-comment"># přepínač -v vypíše v jakých případech máte chybu</span>
(venv) $ python -m pytest -vv tests.py   <span class="hljs-comment"># více v, více verbose</span></code></pre>
</div>

                
            </div>