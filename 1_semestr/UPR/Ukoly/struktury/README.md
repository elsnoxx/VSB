<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak m&#367;&#382;eme v jazyce <em>C</em> nadefinovat sv&#233; vlastn&#237; datov&#233; typy (struktury), co&#382; n&#225;m pom&#367;&#382;e zv&#253;&#353;it abstrakci k&#243;du a umo&#382;n&#237; n&#225;m jej jednodu&#353;&#353;eji pochopit.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/c/struktury/vlastni_datove_typy.html">Vlastn&#237; datov&#233; typy</a>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/struktury/struktury.html">Struktury</a></li>
</ul>
</li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/struktury.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu. Je zak&#225;z&#225;no sd&#237;let sv&#233; &#345;e&#353;en&#237; s ostatn&#237;mi, opisovat od ostatn&#237;ch, nechat si od ostatn&#237;ch diktovat k&#243;d a pou&#382;&#237;vat AI n&#225;stroje na psan&#237; k&#243;du (ChatGPT, Copilot atd.).</strong></p>
<p>Tento t&#253;den si zkus&#237;te napsat program, kter&#253; na&#269;te sadu z&#225;znam&#367; o hodnot&#225;ch akci&#237;. Informace a statistiky o akci&#237;ch pot&#233; program vyp&#237;&#353;e na <strong>standardn&#237; v&#253;stup</strong> ve form&#283; <a href="https://en.wikipedia.org/wiki/HTML">HTML</a> str&#225;nky.</p>
<p>Program by se m&#283;l chovat takto:</p>
<ol type="1">
<li>
<p>Program na&#269;te pomoc&#237; <a href="https://mrlvsb.github.io/upr-skripta/ruzne/funkce_main.html#vstupn%C3%AD-parametry-funkce-main">parametr&#367; p&#345;&#237;kazov&#233; &#345;&#225;dky</a> dva parametry: n&#225;zev akcie (&#345;et&#283;zec <code>t</code>) a po&#269;et &#345;&#225;dk&#367; na vstupu (&#269;&#237;slo <code>n</code>). Nap&#345;.:</p>
<pre class="console"><code>$ ./program AAPL 50</code></pre>
<ul>
<li>Pokud na vstupu programu nebudou oba dva tyto parametry, vypi&#353;te &#345;&#225;dek s hl&#225;&#353;kou <code>Wrong parameters</code> a <a href="https://mrlvsb.github.io/upr-skripta/ruzne/funkce_main.html#funkce-main">ukon&#269;ete</a> program s k&#243;dem <code>1</code>.</li>
<li>
<code>n</code> bude v&#382;dy kladn&#233; &#269;&#237;slo (v&#283;t&#353;&#237; ne&#382; nula).</li>
</ul>
</li>
<li>
<p>D&#225;le program na&#269;te ze vstupu postupn&#283; <code>n</code> &#345;&#225;dk&#367;. Ka&#382;d&#253; &#345;&#225;dek bude obsahovat <strong>z&#225;znam</strong> o hodnot&#283; akcie konkr&#233;tn&#237; firmy v konkr&#233;tn&#237; den. Na &#345;&#225;dku bude p&#283;t &#250;daj&#367; odd&#283;len&#253;ch &#269;&#225;rkou:</p>
<ul>
<li>Index dne (cel&#233; &#269;&#237;slo)</li>
<li>N&#225;zev akcie (&#345;et&#283;zec obsahuj&#237;c&#237; pouze velk&#225; p&#237;smena anglick&#233; abecedy)</li>
<li>Hodnota akcie na za&#269;&#225;tku dan&#233;ho dne (desetinn&#233; &#269;&#237;slo)</li>
<li>Hodnota akcie na konci dan&#233;ho dne (desetinn&#233; &#269;&#237;slo)</li>
<li>Po&#269;et proveden&#253;ch obchod&#367; v dan&#253; den (cel&#233; &#269;&#237;slo)</li>
</ul>
<p>P&#345;&#237;klad <em>z&#225;znamu</em>:</p>
<pre><code>1,NVDA,134.23,135.64,51234158</code></pre>
<ul>
<li>Ka&#382;d&#253; &#345;&#225;dek bude m&#237;t maxim&#225;ln&#283; <code>100</code> znak&#367; (v&#269;etn&#283; znaku od&#345;&#225;dkov&#225;n&#237;).</li>
<li>Ulo&#382;te v pam&#283;ti n&#225;zev akcie v ka&#382;d&#233;m z&#225;znamu tak, aby zab&#237;ral co nejm&#233;n&#283; byt&#367;, tedy pouze tolik byt&#367;, kolik je d&#233;lka n&#225;zvu (plus jeden byte pro <code>NULL</code> termin&#225;tor).</li>
</ul>
</li>
<li>
<p>Nalezn&#283;te mezi zadan&#253;mi &#250;daji v&#353;echny <em>z&#225;znamy</em> akcie <code>t</code>. V nich vyhledejte <em>z&#225;znam</em>, kter&#253; dos&#225;hl nejvy&#353;&#353;&#237;ho po&#269;tu proveden&#253;ch obchod&#367;.</p>
<ul>
<li>Pokud takov&#253; <em>z&#225;znam</em> existuje, vypi&#353;te o n&#283;m dostupn&#233; informace ve form&#225;tu HTML (viz p&#345;&#237;klad n&#237;&#382;e a testy).</li>
<li>Pokud takov&#253; <em>z&#225;znam</em> neexistuje, vypi&#353;te informaci ve form&#225;tu HTML, &#382;e dan&#225; akcie nebyla nalezena (viz testy).</li>
<li>Pokud by <em>z&#225;znam&#367;</em> s nejvy&#353;&#353;&#237;m po&#269;tem obchod&#367; bylo v&#237;ce, vypi&#353;te ten, kter&#253; se ve vstupu vyskytoval d&#345;&#237;ve.</li>
</ul>
</li>
<li>
<p>D&#225;le vypi&#353;te HTML tabulku s ve&#353;ker&#253;mi na&#269;ten&#253;mi vstupn&#237;mi <em>z&#225;znamy</em>, v opa&#269;n&#233;m po&#345;ad&#237;, ne&#382; byly zad&#225;ny na vstupu (tj. nejprve posledn&#237; z&#225;znam, pot&#233; p&#345;edposledn&#237; z&#225;znam atd.). <em>Z&#225;znamy</em> obsahuj&#237;c&#237; informace o akcii <code>t</code> vypi&#353;te tu&#269;n&#283; (obalen&#233; v HTML tagu <code>&lt;b&gt;&lt;/b&gt;</code>). P&#345;&#237;klad vypsan&#233;ho z&#225;znamu:</p>
<pre class="html"><code>&lt;tr&gt;
    &lt;td&gt;26&lt;/td&gt;
    &lt;td&gt;AMD&lt;/td&gt;
    &lt;td&gt;136.92&lt;/td&gt;
    &lt;td&gt;139.17&lt;/td&gt;
    &lt;td&gt;2.25&lt;/td&gt;
    &lt;td&gt;6_471_118&lt;/td&gt;
&lt;/tr&gt;</code></pre>
</li>
</ol>
<p>D&#233;lka referen&#269;n&#237;ho &#345;e&#353;en&#237;: ~160 &#345;&#225;dk&#367; (bez bonusu)</p>
<h3 id="form&#225;t-vypsan&#253;ch-dat">Form&#225;t vypsan&#253;ch dat</h3>
<ul>
<li>Na ka&#382;d&#233;m &#345;&#225;dku tabulky jsou postupn&#283; &#250;daje o indexu dne, n&#225;zvu akcie, hodnot&#283; akcie na za&#269;&#225;tku a konci dne, rozd&#237;l mezi cenou akcie na za&#269;&#225;tku a na konci dan&#233;ho dne, a pot&#233; po&#269;et proveden&#253;ch obchod&#367; v dan&#253; den.</li>
<li>Jednotliv&#233; hodnoty sloupc&#367; vlo&#382;te do HTML tagu <code>&lt;td&gt;</code>, a tento tag ve v&#253;stupu odsa&#271;te tabul&#225;torem.<br>
</li>
<li>V&#353;echny hodnoty po&#269;tu proveden&#253;ch obchod&#367; (jak p&#345;i v&#253;pisu nalezen&#233;ho z&#225;znamu z bodu 3), tak p&#345;i v&#253;pisu tabulky z bodu 4)) by m&#283;ly b&#253;t vyps&#225;ny tak, aby mezi jednotliv&#253;mi trojicemi &#269;&#237;slic byl znak <code>_</code>, aby &#353;lo tyto hodnoty jednodu&#353;&#353;eji &#269;&#237;st.</li>
<li>Ve&#353;ker&#225; desetinn&#225; &#269;&#237;sla vypisujte s p&#345;esnost&#237; na dv&#283; desetinn&#225; m&#237;sta. Jak toho dos&#225;hnout se m&#367;&#382;ete dozv&#283;d&#283;t v <a href="https://devdocs.io/c/io/fprintf">dokumentaci</a> funkce <code>printf</code> (hledejte <em>precision</em>), p&#345;&#237;padn&#283; to m&#367;&#382;ete vygooglit.</li>
</ul>
<h3 id="p&#345;&#237;klad">P&#345;&#237;klad</h3>
<ul>
<li>
<p>Spu&#353;t&#283;n&#237; programu:</p>
<pre class="console"><code>$ ./main TSLA 5</code></pre>
</li>
<li>
<p>Vstup:</p>
<pre class="text"><code>1,TSLA,662.56,664.63,73576275
2,AMD,745.26,749.48,27373290
2,TSLA,664.63,665.66,78968627
3,AMD,749.48,745.39,51760557
3,TSLA,665.66,663.08,24778442</code></pre>
</li>
<li>
<p>Odpov&#237;daj&#237;c&#237; v&#253;stup:</p>
<pre class="html"><code> &lt;html&gt;
 &lt;body&gt;
 &lt;div&gt;
 &lt;h1&gt;TSLA: highest volume&lt;/h1&gt;
 &lt;div&gt;Day: 2&lt;/div&gt;
 &lt;div&gt;Start price: 664.63&lt;/div&gt;
 &lt;div&gt;End price: 665.66&lt;/div&gt;
 &lt;div&gt;Volume: 78_968_627&lt;/div&gt;
 &lt;/div&gt;
 &lt;table&gt;
 &lt;thead&gt;
 &lt;tr&gt;&lt;th&gt;Day&lt;/th&gt;&lt;th&gt;Ticker&lt;/th&gt;&lt;th&gt;Start&lt;/th&gt;&lt;th&gt;End&lt;/th&gt;&lt;th&gt;Diff&lt;/th&gt;&lt;th&gt;Volume&lt;/th&gt;&lt;/tr&gt;
 &lt;/thead&gt;
 &lt;tbody&gt;
 &lt;tr&gt;
         &lt;td&gt;&lt;b&gt;3&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;TSLA&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;665.66&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;663.08&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;-2.58&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;24_778_442&lt;/b&gt;&lt;/td&gt;
 &lt;/tr&gt;
 &lt;tr&gt;
         &lt;td&gt;3&lt;/td&gt;
         &lt;td&gt;AMD&lt;/td&gt;
         &lt;td&gt;749.48&lt;/td&gt;
         &lt;td&gt;745.39&lt;/td&gt;
         &lt;td&gt;-4.09&lt;/td&gt;
         &lt;td&gt;51_760_557&lt;/td&gt;
 &lt;/tr&gt;
 &lt;tr&gt;
         &lt;td&gt;&lt;b&gt;2&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;TSLA&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;664.63&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;665.66&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;1.03&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;78_968_627&lt;/b&gt;&lt;/td&gt;
 &lt;/tr&gt;
 &lt;tr&gt;
         &lt;td&gt;2&lt;/td&gt;
         &lt;td&gt;AMD&lt;/td&gt;
         &lt;td&gt;745.26&lt;/td&gt;
         &lt;td&gt;749.48&lt;/td&gt;
         &lt;td&gt;4.22&lt;/td&gt;
         &lt;td&gt;27_373_290&lt;/td&gt;
 &lt;/tr&gt;
 &lt;tr&gt;
         &lt;td&gt;&lt;b&gt;1&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;TSLA&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;662.56&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;664.63&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;2.07&lt;/b&gt;&lt;/td&gt;
         &lt;td&gt;&lt;b&gt;73_576_275&lt;/b&gt;&lt;/td&gt;
 &lt;/tr&gt;
 &lt;/tbody&gt;
 &lt;/table&gt;
 &lt;/body&gt;
 &lt;/html&gt;</code></pre>
</li>
</ul>
<p>Pokud si v&#253;stup programu p&#345;esm&#283;rujete do souboru s koncovkou <code>.html</code>, tak si jej pot&#233; m&#367;&#382;ete otev&#345;&#237;t ve webov&#233;m prohl&#237;&#382;e&#269;i a pod&#237;vat se, jak bude v&#253;stup graficky vypadat:</p>
<pre class="console"><code>$ ./main TSLA 5 &lt; test-small.stdin &gt; output.html</code></pre>
<p>Pln&#233; uk&#225;zky v&#253;stupu si m&#367;&#382;ete prohl&#233;dnout v z&#225;lo&#382;ce <a href="#tests">Tests</a>.</p>
<h3 id="pozn&#225;mky">Pozn&#225;mky</h3>
<ul>
<li>
<strong>Pou&#382;it&#237; <a href="https://mrlvsb.github.io/upr-skripta/c/pole/staticka_pole.html#konstantn%C3%AD-velikost-statick%C3%A9ho-pole">VLA</a> je zak&#225;z&#225;no</strong>.</li>
<li><strong>V implementaci &#250;lohy vhodn&#283; vyu&#382;ijte struktury a nadefinujte si vlastn&#237; datov&#253;(&#233;) typ(y)!</strong></li>
<li>Pro reprezentaci desetinn&#253;ch &#269;&#237;sel pou&#382;ijte datov&#253; typ <code>float</code>.</li>
<li>D&#225;vejte si u funkce <code>fgets</code> pozor na to, &#382;e znak od&#345;&#225;dkov&#225;n&#237; je tak&#233; sou&#269;&#225;st&#237; vstupu! Viz <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html#na%C4%8Dten%C3%AD-%C5%99%C3%A1dku">skripta</a>.</li>
<li>P&#345;i pr&#225;ci s &#345;et&#283;zci budete nar&#225;&#382;et na pam&#283;&#357;ov&#233; chyby. Pou&#382;&#237;vejte <a href="#kontrola-pam%C4%9B%C5%A5ov%C3%BDch-chyb">Address sanitizer nebo Valgrind</a>! P&#345;i &#345;e&#353;en&#237; t&#233;to &#250;lohy bude velmi u&#382;ite&#269;n&#233; vyu&#382;&#237;t <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#krokov%C3%A1n%C3%AD">debugger</a> VSCode. I p&#345;i lad&#283;n&#237;/krokov&#225;n&#237; si m&#367;&#382;ete na vstup programu <a href="https://mrlvsb.github.io/upr-skripta/prostredi/editor.html#pokro%C4%8Dil%C3%A9-mo%C5%BEnosti-nastaven%C3%AD-projektu">p&#345;esm&#283;rovat</a> soubor, abyste nemuseli vstup neust&#225;le ps&#225;t ru&#269;n&#283;.</li>
</ul>
<h3 id="u&#382;ite&#269;n&#233;-funkce">U&#382;ite&#269;n&#233; funkce</h3>
<ul>
<li>
<a href="https://devdocs.io/c/io/fgets"><code>fgets</code></a> - na&#269;ten&#237; &#345;&#225;dku ze vstupu do &#345;et&#283;zce (pole znak&#367;).</li>
<li>
<a href="https://devdocs.io/c/string/byte/atoi"><code>atoi</code></a> - p&#345;eveden&#237; &#345;et&#283;zce obsahuj&#237;c&#237;ho &#269;&#237;slice na cel&#233; &#269;&#237;slo (<code>int</code>).</li>
<li>
<a href="https://devdocs.io/c/string/byte/atof"><code>atof</code></a> - p&#345;eveden&#237; &#345;et&#283;zce obsahuj&#237;c&#237;ho &#269;&#237;slice na desetinn&#237; &#269;&#237;slo (<code>float</code>).</li>
<li>
<a href="https://devdocs.io/c/string/byte/strlen"><code>strlen</code></a> - zji&#353;t&#283;n&#237; d&#233;lky &#345;et&#283;zce.</li>
<li>
<a href="https://devdocs.io/c/string/byte/strtok"><code>strtok</code></a> - rozd&#283;len&#237; &#345;et&#283;zce nap&#345;. podle mezer. <em>Tato funkce je trochu komplikovan&#225;, m&#367;&#382;e b&#253;t pro v&#225;s jednodu&#353;&#353;&#237; si rozd&#283;len&#237; podle mezer naprogramovat &#8220;ru&#269;n&#283;&#8221;.</em>
</li>
</ul>
<h3 id="bonusov&#253;-&#250;kol">Bonusov&#253; &#250;kol</h3>
<p>Vykreslete do separ&#225;tn&#237;ho souboru i graf (nap&#345;. jednoduch&#253; sv&#237;&#269;kov&#253;, nebo sloupcov&#253;) ve form&#225;tu SVG, kter&#253; bude zn&#225;zor&#328;ovat v&#253;voj hodnoty akcie <code>t</code> v &#269;ase. P&#345;&#237;klad:</p>
<svg width="800" height="600" viewbox="0 0 800 600" xlmns="http://www.w3.org/2000/svg">
<polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="30.769232, 210.429016 30.769232, 166.214661"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="61.538464, 166.214661 61.538464, 132.419373"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="123.076927, 132.419373 123.076927, 164.690094"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="153.846161, 164.690094 153.846161, 231.011261"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="184.615387, 231.011261 184.615387, 130.640472"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="215.384628, 130.640472 215.384628, 148.173615"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="246.153854, 148.173615 246.153854, 102.435455"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="307.692322, 102.435455 307.692322, 97.861023"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="369.230774, 97.861023 369.230774, 60.000000"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="430.769257, 60.000000 430.769257, 81.344482"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="461.538452, 81.344482 461.538452, 155.542816"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="523.076904, 155.542816 523.076904, 261.249542"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="553.846191, 261.249542 553.846191, 255.404907"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="584.615356, 255.404907 584.615356, 372.037842"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="615.384644, 372.037842 615.384644, 339.767090"></polyline> <polyline fill="none" stroke="#00FF00" stroke-width="10.000000" points="646.153809, 339.767090 646.153809, 301.143005"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="707.692261, 301.143005 707.692261, 415.489136"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="738.461548, 415.489136 738.461548, 477.744568"></polyline> <polyline fill="none" stroke="#FF0000" stroke-width="10.000000" points="769.230774, 477.744568 769.230774, 540.000000"></polyline>
</svg>
<p>Pokud budete bonusov&#253; &#250;kol implementovat, nechejte ho v k&#243;du zakomentovan&#253;, a&#357; projdou testy na Kelvinovi, a p&#345;idejte do Kelvina koment&#225;&#345; pro sv&#233;ho cvi&#269;&#237;c&#237;ho, &#382;e m&#225;te bonus hotov&#253;.</p>
<h3 id="testov&#225;n&#237;-programu">Testov&#225;n&#237; programu</h3>
<p>Uk&#225;zkov&#233; vstupy a v&#253;stupy naleznete v z&#225;lo&#382;ce <code>Tests</code>. Odtud si je tak&#233; m&#367;&#382;ete st&#225;hnout (<code>stdin</code> - vstup, <code>stdout</code> - o&#269;ek&#225;v&#225;n&#253; v&#253;stup z va&#353;eho programu). Nezapome&#328;te tak&#233; spr&#225;vn&#283; p&#345;ed&#225;vat va&#353;emu programu parametry p&#345;&#237;kazov&#233;ho &#345;&#225;dku!</p>
<p>Po nahr&#225;n&#237; zdrojov&#233;ho souboru se m&#367;&#382;ete pod&#237;vat, jestli testy pro&#353;ly nebo ne. To, &#382;e v&#353;echny testy pro&#353;ly, v&#353;ak je&#353;t&#283; neznamen&#225;, &#382;e je v&#225;&#353; program spr&#225;vn&#283; :) Stejn&#283; tak naopak, pokud v&#353;echny testy nepro&#353;ly, neznamen&#225; to automaticky, &#382;e m&#225;te nula bod&#367;.</p>
<h3 id="p&#345;esm&#283;rov&#225;n&#237;-vstupu">P&#345;esm&#283;rov&#225;n&#237; vstupu</h3>
<p>Abyste nemuseli neust&#225;le ru&#269;n&#283; zad&#225;vat &#269;&#237;sla z kl&#225;vesnice p&#345;i testov&#225;n&#237; programu, m&#367;&#382;ete data na vstup programu <a href="https://www.pslib.cz/milan.kerslager/BASH:_P%C5%99esm%C4%9Brov%C3%A1n%C3%AD">p&#345;esm&#283;rovat</a> ze souboru:</p>
<pre class="sh"><code># p&#345;eklad programu
gcc -g -fsanitize=address main.c -o program

# spu&#353;t&#283;n&#237; souboru, p&#345;esm&#283;rov&#225;n&#237; souboru test-small.stdin na vstup programu
./program TSLA 5 &lt; test-small/stdin</code></pre>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
</div>
