<div>
<p>Ukážeme si, jak pracovat se třídami a moduly.</p>
<p>Odpovídající materiály naleznete <a href="https://naucse.python.cz/course/pyladies/sessions/class/">zde</a> a <a href="https://naucse.python.cz/course/pyladies/sessions/tests/">zde</a>.</p>
<h2>Materiály:</h2>
<ul>
<li>
<a href="/task/SKJ/2024S/GAU01/w04_oop/asset/template/cheatsheet.py">cheatsheet.py</a> - tahák s třídami.</li>
<li>
<a href="/task/SKJ/2024S/GAU01/w04_oop/asset/template/tasks.py">tasks.py</a> - zadání na cvičení a DÚ</li>
<li>
<a href="/task/SKJ/2024S/GAU01/w04_oop/asset/template/tests.py">tests.py</a> - testy ke cvičení a DÚ</li>
</ul>
<p>Slidy naleznete <a href="/task/SKJ/2024S/GAU01/w04_oop/asset/assets/slides.pdf">zde</a>.</p>
<h2>Cvičení</h2>
<p>Naimplementujte třídy <code>Vector</code> a <code>Observable</code> (3 body).</p>
<h2>DÚ</h2>
<p>Doimplementujte třídy <code>UpperCaseDecorator</code> a <code>GameOfLife</code> (2 body) a odevzdejte v den zadání do 23:59
(na pozdější odevzdání nebude brán ohled).</p>
<p>Řešení odevzdávejte zde do Kelvinu (stačí odevzdat vyřešený soubor <code>tasks.py</code>).</p>
<p><strong>Na úloze pracujte samostatně bez použití AI nástrojů!</strong></p>
<h2>Nastavení prostředí a testů</h2>
<p>Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).
Nainstalujte si do něj knihovnu <code>pytest</code>, abyste mohli spouštět připravené unit testy.
</p>
<pre class="highlight"><code class="language-bash hljs" data-highlighted="yes">$ python3 -m venv venv/skj          <span class="hljs-comment"># Vytvoří virtuální prostředí pro instalaci balíčků (spusťte pouze jednou)</span>
$ <span class="hljs-built_in">source</span> venv/skj/bin/activate      <span class="hljs-comment"># Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)</span>
(skj) $ pip install pytest          <span class="hljs-comment"># Nainstaluje balíček pip do virtuálního prostředí (spusťte pouze jednou)</span>
(skj) $ python -m pytest tests.py   <span class="hljs-comment"># Spustí testy ze souboru tests.py</span></code></pre>
</div>