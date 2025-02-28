<div class="tab-pane active" id="tab_assignment">
                <div>
<p>Tento týden máte za úkol zejména nastavit si prostředí pro vývoj Pythonu na Linuxu, seznámit se
se základní syntaxí Pythonu a pochopit, jak funguje jeho správa paměti a interpretace.</p>
<p>Odpovídá to cca prvním třem lekcím <a href="https://naucse.python.cz/course/pyladies/">kurzu PyLadies</a>.</p>
<p>Pro vývoj v Pythonu doporučuju jako IDE PyCharm nebo VSCode.</p>
<h2>Materiály:</h2>
<ul>
<li>
<a href="/task/SKJ/2022S/BER0134/ex01_basics/asset/assets/cheatsheet.py">cheatsheet.py</a> - tahák se základy syntaxe Pythonu</li>
<li>
<a href="/task/SKJ/2022S/BER0134/ex01_basics/asset/template/tasks.py">tasks.py</a> - zadání DÚ</li>
<li>
<a href="/task/SKJ/2022S/BER0134/ex01_basics/asset/template/tests.py">tests.py</a> - testy k DÚ</li>
</ul>
<h2>DÚ</h2>
<p>Řešení domácí úlohy odevzdávejte zde do Kelvina.
<strong>Odevzdejte pouze jeden soubor s vyřešenými úlohami, pojmenovaný <code>tasks.py</code>.</strong>
Za úlohy můžete získat maximálně 3 body.</p>
<h2>Nastavení prostředí a testů</h2>
<p>Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).
Nainstalujte si do něj knihovnu <code>pytest</code>, abyste mohli spouštět připravené unit testy.
</p>
<pre class="highlight"><code class="language-bash hljs" data-highlighted="yes">$ python3 -m venv venv/skj          <span class="hljs-comment"># Vytvoří virtuální prostředí pro instalaci balíčků (spusťte pouze jednou)</span>
$ <span class="hljs-built_in">source</span> venv/skj/bin/activate      <span class="hljs-comment"># Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)</span>
(skj) $ pip install pytest          <span class="hljs-comment"># Nainstaluje balíček pip do virtuálního prostředí (spusťte pouze jednou)</span>
(skj) $ python -m pytest tests.py   <span class="hljs-comment"># Spustí testy ze souboru tests.py</span></code></pre>
</div>

                
</div>