<div class="tab-pane active" id="tab_assignment">
                <div>
<p>Stáhněte si archiv <a href="/task/SKJ/2024S/GAU01/zadani_cv07_game/asset/template.zip">template.zip</a>, kde naleznete herní server a herního klienta.
Upravujete pouze soubor <code>client.py</code>. Herní server bude běžet na počítači cvičícího, abyste se případně viděli všichni ve výsledné hře.
Postupujte dle zadání.</p>
<p>Odevzdávejte soubor s klientem.</p>
<h2>Nastavení prostředí</h2>
<p>Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).
Nainstalujte si do něj knihovnu <code>pygame</code>, abyste mohli spouštět připravený template.
</p>
<pre class="highlight"><code class="language-bash hljs" data-highlighted="yes">$ python3 -m venv venv/skj          <span class="hljs-comment"># Vytvoří virtuální prostředí pro instalaci balíčků (spusťte pouze jednou)</span>
$ <span class="hljs-built_in">source</span> venv/skj/bin/activate      <span class="hljs-comment"># Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)</span>
(skj) $ pip install pygame          <span class="hljs-comment"># Nainstaluje balíček pip do virtuálního prostředí (spusťte pouze jednou)</span>
(skj) $ python client.py            <span class="hljs-comment"># Spustí aplikaci</span></code></pre>
</div>

                
</div>