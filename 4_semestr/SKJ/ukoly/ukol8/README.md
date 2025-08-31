<div class="tab-pane active" id="tab_assignment">
                <div>
<h2>Materiály:</h2>
<ul>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/template/tasks.py">tasks.py</a> - zadání</li>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/template/tests.py">tests.py</a> - testy</li>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/assets/skj_stack.pdf">skj_stack.pdf</a> - prezentace</li>
</ul>
<h2>Odevzdání</h2>
<p>Řešení úlohy odevzdávejte zde do Kelvinu (odevzdejte vyřešený soubor <code>tasks.py</code>).
Za každý správně naimplementovaný test dostanete 1 bod, tedy maximálně můžete získat 5 bodů.</p>
<p>V případě, že neodevzdáte do konce hodiny,
můžete úlohu dodělat doma a odevzdat do večera 23:59.</p>
<p>Podmínkou pro udělení bodů za vypracování doma je účast na hodině. Jen v případě, že se vám to nepodaří na hodině, tak můžete dodělat doma.</p>
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