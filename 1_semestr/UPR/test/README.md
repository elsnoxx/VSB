<div>
<p>Verze zad&#225;n&#237; pro: <strong>FIC0024</strong></p>
<p><strong>Hled&#225; se gul&#225;&#353;!</strong> Zastupitel akademick&#233; obce sl&#237;bil student&#367;m FEI po&#345;&#225;dnou porci gul&#225;&#353;e jako odm&#283;nu za &#250;&#269;ast ve volb&#225;ch. Jak u&#382; to tak ale b&#253;v&#225;, tento slib nebyl dodr&#382;en, a tak si mus&#237;me gul&#225;&#353; uva&#345;it sami. Uva&#345;it gul&#225;&#353; ale nen&#237; jen tak! Ka&#382;d&#225; katedra m&#225; sv&#367;j vlastn&#237; recept na gul&#225;&#353;, kter&#253; by cht&#283;la nechat uva&#345;it, ale z&#225;soby ingredienc&#237; menzy jsou omezen&#233;. Mus&#237;me tak vybrat, kter&#253; recept pou&#382;ijeme na uva&#345;en&#237; gul&#225;&#353;e, tak, abychom zasytili co nejv&#237;ce student&#367;.</p>
<p>Pomozte tuto situaci vy&#345;e&#353;it t&#237;m, &#382;e vytvo&#345;&#237;te program, kter&#253; obdr&#382;&#237; na vstupu seznam dostupn&#253;ch <strong>ingredienc&#237;</strong> a seznam <strong>recept&#367;</strong>, a zjist&#237;, z kter&#233;ho receptu lze uva&#345;it nejv&#237;ce porc&#237; gul&#225;&#353;e.</p>
<p>Krom&#283; popisu fungov&#225;n&#237; programu se tak&#233; d&#367;kladn&#283; pod&#237;vejte na <a href="#pozn%C3%A1mky-k-zad%C3%A1n%C3%AD-a-%C5%99e%C5%A1en%C3%AD">pozn&#225;mky</a> k zad&#225;n&#237;. <strong>Pokud &#269;emukoliv v zad&#225;n&#237; nerozum&#237;te, zeptejte se ihned zkou&#353;ej&#237;c&#237;ho dozoru!</strong></p>
<h2 id="popis-fungov&#225;n&#237;-programu">Popis fungov&#225;n&#237; programu</h2>
<ol type="1">
<li>
<p>V&#225;&#353; program obdr&#382;&#237; pomoc&#237; <a href="https://mrlvsb.github.io/upr-skripta/ruzne/funkce_main.html#vstupn%C3%AD-parametry-funkce-main">parametr&#367; p&#345;&#237;kazov&#233;ho &#345;&#225;dku</a> dv&#283; celo&#269;&#237;seln&#233; nez&#225;porn&#233; hodnoty:</p>
<ul>
<li>Po&#269;et &#345;&#225;dk&#367; s <strong>ingrediencemi</strong> (<code>i</code>).</li>
<li>Po&#269;et &#345;&#225;dk&#367; s <strong>recepty</strong> (<code>r</code>).</li>
</ul>
<p>P&#345;&#237;klad pro <code>5</code> &#345;&#225;dk&#367; s ingrediencemi a <code>2</code> &#345;&#225;dky s recepty:</p>
<pre class="console"><code>$ ./program 5 2</code></pre>
<ul>
<li>Po&#269;et ingredienc&#237; i recept&#367; bude nez&#225;porn&#233; &#269;&#237;slo, tj. bu&#271; <code>0</code>, nebo kladn&#233; &#269;&#237;slo.</li>
<li>Nemus&#237;te kontrolovat, jestli byly parametry p&#345;ed&#225;ny, m&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e program v&#382;dy oba dva tyto parametry obdr&#382;&#237;.</li>
</ul>
</li>
<li>
<p>Ze standardn&#237;ho vstupu na&#269;t&#283;te <code>i</code> &#345;&#225;dk&#367;. Na ka&#382;d&#233;m &#345;&#225;dku budou n&#225;sleduj&#237;c&#237; &#250;daje o jedn&#233; <strong>ingredienci</strong>, kter&#225; je k dispozici pro va&#345;en&#237;, odd&#283;len&#233; &#269;&#225;rkou:</p>
<ul>
<li>
<strong>N&#225;zev</strong> (&#345;et&#283;zec, max. 100 znak&#367;, nebude obsahovat &#269;&#225;rku)</li>
<li>
<strong>Po&#269;et</strong> (kladn&#233; cel&#233; &#269;&#237;slo v&#283;t&#353;&#237; ne&#382; <code>0</code>)</li>
</ul>
<p>Nap&#345;&#237;klad:</p>
<pre><code>Paprika,50</code></pre>
<p>ud&#225;v&#225;, &#382;e m&#225;te k dispozici 50 kus&#367; ingredience s n&#225;zvem <code>Paprika</code>.</p>
<p>Ulo&#382;te si informace o ingredienc&#237;ch do pole v pam&#283;ti.</p>
<ul>
<li>N&#225;zvy ingredienc&#237; ve vstupu budou unik&#225;tn&#237;. Jinak &#345;e&#269;eno, v seznamu ingredienc&#237; se nikdy nebude ingredience se stejn&#253;m n&#225;zvem vyskytovat v&#237;ce, ne&#382; jednou.</li>
<li>V&#225;&#353; program mus&#237; fungovat pro libovoln&#253; po&#269;et ingredienc&#237;! Nesta&#269;&#237; si ur&#269;it natvrdo konkr&#233;tn&#237; maxim&#225;ln&#237; velikost pole s ingrediencemi v dob&#283; p&#345;ekladu.</li>
</ul>
</li>
<li>
<p>Ze standardn&#237;ho vstupu d&#225;le na&#269;t&#283;te <code>r</code> &#345;&#225;dk&#367;. Na ka&#382;d&#233;m &#345;&#225;dku bude jeden <strong>recept</strong> na gul&#225;&#353;, kter&#253; se bude skl&#225;dat ze sady ingredienc&#237;. Form&#225;t &#345;&#225;dku s receptem je n&#225;sleduj&#237;c&#237;:</p>
<pre><code>&lt;n&gt;;&lt;ingredience 1 n&#225;zev&gt;,&lt;ingredience 1 po&#269;et&gt;;&lt;ingredience 2 n&#225;zev&gt;,&lt;ingredience 2 po&#269;et&gt;;...</code></pre>
<p>Na za&#269;&#225;tku &#345;&#225;dku je kladn&#233; &#269;&#237;slo (<code>n</code>) ud&#225;vaj&#237;c&#237; po&#269;et jednotliv&#253;ch ingredienc&#237;, ze kter&#253;ch se recept skl&#225;d&#225;. Po n&#283;m n&#225;sleduje st&#345;edn&#237;k (<code>;</code>), a d&#225;le <code>n</code> ingredienc&#237; odd&#283;len&#253;ch st&#345;edn&#237;kem. Ka&#382;d&#225; ingredience v receptu m&#225; stejn&#253; form&#225;t jako v seznamu ingredienc&#237; na za&#269;&#225;tku vstupu, tj. n&#225;zev ingredience n&#225;sledovan&#253; &#269;&#225;rkou a pot&#233; po&#269;tem kus&#367; dan&#233; ingredience, kter&#253; je v receptu vy&#382;adov&#225;n.</p>
<p>P&#345;&#237;klad:</p>
<pre><code>3;Coconut oil,2;Tortillas,1;Paprika,5</code></pre>
<p>Tento &#345;&#225;dek ud&#225;v&#225;, &#382;e recept vy&#382;aduje t&#345;i ingredience. Recept se skl&#225;d&#225; ze dvou kus&#367; ingredience <code>Coconut oil</code>, jednoho kusu ingredience <code>Tortillas</code> a z p&#283;ti kus&#367; ingredience <code>Paprika</code>.</p>
<ul>
<li>N&#225;zvy ingredienc&#237; v receptu budou unik&#225;tn&#237;. Jinak &#345;e&#269;eno, v seznamu ingredienc&#237; receptu se nikdy nebude ingredience se stejn&#253;m n&#225;zvem vyskytovat v&#237;ce, ne&#382; jednou.</li>
<li>Ka&#382;d&#253; recept bude obsahovat alespo&#328; jednu ingredienci, tj. <code>n</code> bude v&#382;dy v&#283;t&#353;&#237;, ne&#382; <code>0</code>.</li>
<li>Po&#269;et ka&#382;d&#233; ingredience v receptu bude kladn&#253;, tj. ka&#382;d&#225; ingredience bude v&#382;dy m&#237;t po&#269;et v&#283;t&#353;&#237;, ne&#382; <code>0</code>.</li>
</ul>
</li>
<li>
<p>Pro ka&#382;d&#253; recept ve vstupu zjist&#283;te, kolik <strong>porc&#237;</strong> tohoto receptu lze uva&#345;it pomoc&#237; dostupn&#253;ch ingredienc&#237;. Pot&#233; nalezn&#283;te <strong>nejlep&#353;&#237; recept</strong>. Nejlep&#353;&#237; recept je ten, ze kter&#233;ho lze uva&#345;it nejv&#237;ce porc&#237; gul&#225;&#353;e.</p>
<p>Pro v&#253;po&#269;et po&#269;tu porc&#237; receptu, kter&#233; lze uva&#345;it, zjist&#283;te, kolik kus&#367; ka&#382;d&#233; ingredience z receptu m&#225;te k dispozici (viz seznam ingredienc&#237; na za&#269;&#225;tku vstupu). Pot&#233; vypo&#269;&#237;tejte, kolik porc&#237; receptu m&#367;&#382;ete uva&#345;it. Pro ka&#382;dou porci mus&#237;te m&#237;t k dispozici v&#353;echny ingredience receptu v zadan&#233;m po&#269;tu. Berte v potaz pouze cel&#233; porce. Nap&#345;&#237;klad, pro tento vstup:</p>
<pre><code>Banana,13
Pizza,6
Sausage,7
3;Sausage,3;Banana,2;Pizza,6</code></pre>
<p>M&#225;te k dispozici <code>13x</code> ingredienci <code>Banana</code>, <code>6x</code> ingredienci <code>Pizza</code> a <code>7x</code> ingredienci <code>Sausage</code>. D&#225;le je ve vstupu jedin&#253; recept. Tento recept vy&#382;aduje:</p>
<ul>
<li>
<code>3x</code> <code>Sausage</code>. T&#233;to ingredience je k dispozici <code>7</code> kus&#367;, tak&#382;e pouze vzhledem k t&#233;to ingredienci m&#367;&#382;ete uva&#345;it maxim&#225;ln&#283; dv&#283; porce.</li>
<li>
<code>2x</code> <code>Banana</code>. T&#233;to ingredience je k dispozici <code>13</code> kus&#367;, tak&#382;e pouze vzhledem k t&#233;to ingredienci m&#367;&#382;ete uva&#345;it maxim&#225;ln&#283; &#353;est porc&#237;.</li>
<li>
<code>6x</code> <code>Pizza</code>. T&#233;to ingredience je k dispozici <code>6</code> kus&#367;, tak&#382;e pouze vzhledem k t&#233;to ingredienci m&#367;&#382;ete uva&#345;it maxim&#225;ln&#283; jednu porci.</li>
</ul>
<p>T&#237;m p&#225;dem tohoto receptu m&#367;&#382;ete uva&#345;it maxim&#225;ln&#283; jednu porci, proto&#382;e na v&#237;ce porc&#237; byste u&#382; nem&#283;li k dispozici ingredienci <code>Pizza</code>.</p>
<p>Na standardn&#237; v&#253;stup pot&#233; vypi&#353;te jeden &#345;&#225;dek s <em>indexem</em> nejlep&#353;&#237;ho receptu, a <em>po&#269;tem</em> porc&#237;, kter&#233; z n&#283;j lze uva&#345;it:</p>
<pre><code>Recept &lt;index&gt; lze uvarit &lt;pocet&gt;x</code></pre>
<ul>
<li>
<em>Index</em> receptu je d&#225;n po&#345;ad&#237;m dan&#233;ho receptu ve vstupu, a po&#269;&#237;t&#225;me jej od <code>1</code>, tj. prvn&#237; recept m&#225; index <code>1</code>, druh&#253; <code>2</code> atd.</li>
<li>Pokud by bylo v&#237;ce nejlep&#353;&#237;ch recept&#367;, tak z nich vypi&#353;te recept s nejni&#382;&#353;&#237;m indexem.</li>
<li>Pokud nelze uva&#345;it ani jeden recept, vypi&#353;te m&#237;sto toho &#345;&#225;dek <code>Zadny recept nelze uvarit :(</code>.</li>
</ul>
<p>P&#345;&#237;klad:</p>
<pre><code>Banana,13
Pizza,6
Sausage,7
3;Onion,5;Banana,1;Pizza,1
2;Pizza,2;Banana,4
3;Sausage,1;Banana,4;Pizza,4
1;Sausage,2</code></pre>
<ul>
<li>Recept s indexem <code>1</code> nelze uva&#345;it ani jednou, proto&#382;e nem&#225;me k dispozici ingredienci <code>Onion</code>.</li>
<li>Recept s indexem <code>2</code> lze uva&#345;it <code>3x</code>.</li>
<li>Recept s indexem <code>3</code> lze uva&#345;it <code>1x</code>.</li>
<li>Recept s indexem <code>4</code> lze uva&#345;it <code>3x</code>.</li>
</ul>
<p>Nejlep&#353;&#237; recepty jsou ty s indexy <code>2</code> a <code>4</code>, tak&#382;e program by m&#283;l vypsat, &#382;e &#8220;vyhr&#225;l&#8221; recept s indexem <code>2</code>:</p>
<pre><code>Recept 2 lze uvarit 3x</code></pre>
</li>
</ol>
<p>Kompletn&#237; uk&#225;zkov&#233; vstupy a v&#253;stupy naleznete v z&#225;lo&#382;ce <a href="#tests"><code>Tests</code></a>.</p>
<h3 id="pozn&#225;mky-k-zad&#225;n&#237;-a-&#345;e&#353;en&#237;">Pozn&#225;mky k zad&#225;n&#237; a &#345;e&#353;en&#237;</h3>
<ul>
<li>
<strong>Pou&#382;it&#237; <a href="https://mrlvsb.github.io/upr-skripta/c/pole/staticka_pole.html#konstantn%C3%AD-velikost-statick%C3%A9ho-pole">VLA</a> je zak&#225;z&#225;no</strong>.</li>
<li>
<strong>Nekombinujte funkce <code>scanf</code> a <code>fgets</code> (viz <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html#zpracov%C3%A1n%C3%AD-b%C3%ADl%C3%BDch-znak%C5%AF">skripta</a>), m&#367;&#382;e to zp&#367;sobovat probl&#233;my. Ve&#353;ker&#233; na&#269;&#237;t&#225;n&#237; textu prov&#225;d&#283;jte pomoc&#237; funkce <code>fgets</code>.</strong>
<ul>
<li>Ve&#353;ker&#253; text bude ve form&#225;tu ASCII.</li>
<li>&#381;&#225;dn&#253; &#345;&#225;dek na vstupu nebude m&#237;t v&#237;ce ne&#382; <code>100</code> znak&#367; (v&#269;etn&#283; znaku od&#345;&#225;dkov&#225;n&#237;).</li>
<li>N&#225;zvy ingredienc&#237; si m&#367;&#382;ete ukl&#225;dat v poli znak&#367; s fixn&#237; velikost&#237;, m&#367;&#382;e to b&#253;t jednodu&#353;&#353;&#237;, ne&#382; je alokovat na hald&#283;.</li>
<li>P&#345;i na&#269;&#237;t&#225;n&#237; vstupu nemus&#237;te kontrolovat chyby p&#345;i &#269;ten&#237;, tj. m&#367;&#382;ete o&#269;ek&#225;vat, &#382;e vstup programu bude v&#382;dy ve validn&#237;m form&#225;tu, kter&#253; byl pops&#225;n v&#253;&#353;e.</li>
<li>D&#225;vejte si pozor na existenci znaku od&#345;&#225;dkov&#225;n&#237; v na&#269;ten&#233;m &#345;et&#283;zci!</li>
</ul>
</li>
<li>P&#345;i lok&#225;ln&#237;m testov&#225;n&#237; pou&#382;&#237;vejte <a href="#testov%C3%A1n%C3%AD-programu">p&#345;esm&#283;rov&#225;n&#237; vstupu</a>, abyste nemuseli neust&#225;le ps&#225;t vstup programu do termin&#225;lu.</li>
<li>Va&#353;e &#345;e&#353;en&#237; mus&#237; b&#253;t naps&#225;no v jazyce <em>C</em>. Nebojte se vyu&#382;&#237;vat funkce pro pr&#225;ci s &#345;et&#283;zci a pam&#283;t&#237; ze standardn&#237; knihovny jazyka <em>C</em>.</li>
<li>Ujist&#283;te se, &#382;e ve va&#353;em programu nejsou pam&#283;&#357;ov&#233; chyby. V&#225;&#353; program mus&#237; fungovat se zapnut&#253;m Address Sanitizerem, kter&#253; by nem&#283;l ozn&#225;mit &#382;&#225;dn&#233; chyby.</li>
<li>Rozumn&#233; &#345;e&#353;en&#237; by se m&#283;lo vej&#237;t do ~100&#8212;150 &#345;&#225;dk&#367;. Nevym&#253;&#353;lejte slo&#382;itosti a rad&#353;i se zeptejte, kdyby v&#225;m n&#283;co nebylo jasn&#233;.</li>
<li>Referen&#269;n&#237; v&#253;sledky test&#367; mohou ve v&#253;jime&#269;n&#253;ch p&#345;&#237;padech obsahovat chybu. V p&#345;&#237;pad&#283; pochyb se v&#382;dy &#345;i&#271;te zad&#225;n&#237;m nebo se pora&#271;te se cvi&#269;&#237;c&#237;m, testy jsou pouze pro kontrolu.</li>
</ul>
<h2 id="d&#367;le&#382;it&#233;-informace">D&#367;le&#382;it&#233; informace</h2>
<ul>
<li>Zdrojov&#253; soubor s va&#353;&#237;m k&#243;dem odevzd&#225;vejte do Kelvina. Soubor mus&#237; m&#237;t koncovku <code>.c</code> a mus&#237; j&#237;t p&#345;elo&#382;it p&#345;eklada&#269;em <code>GCC</code> na Linuxu.</li>
<li>Na &#250;kol m&#225;te <code>105</code> minut.</li>
<li>M&#367;&#382;ete z&#237;skat maxim&#225;ln&#283; <code>15</code> bod&#367;.</li>
<li>B&#283;hem testu m&#367;&#382;ete pou&#382;&#237;vat pouze <a href="https://devdocs.io/c/">dokumentaci C</a> a <a href="https://mrlvsb.github.io/upr-skripta/">skripta UPR</a>. Googlen&#237;, hled&#225;n&#237; &#345;e&#353;en&#237; na Stack Overflow atd. je zak&#225;z&#225;no. Pou&#382;&#237;v&#225;n&#237; jin&#253;ch komunika&#269;n&#237;ch prost&#345;edk&#367; (mobiln&#237; telefon, chytr&#233; hodinky, Discord, Teams, Messenger atd.) je zak&#225;z&#225;no. Pou&#382;&#237;v&#225;n&#237; jak&#253;chkoliv AI technologi&#237; je zak&#225;z&#225;no. Pokud by v&#225;m n&#283;co v zad&#225;n&#237; nebylo jasn&#233;, zeptejte se sv&#233;ho cvi&#269;&#237;c&#237;ho.</li>
<li><strong>Pokud Kelvin nebo vyu&#269;uj&#237;c&#237; nalezne opsan&#233; &#345;e&#353;en&#237; nebo jin&#253; pokus o podvod p&#345;i psan&#237; testu, v&#353;ichni studenti s identick&#253;m nebo extr&#233;mn&#283; podobn&#253;m &#345;e&#353;en&#237;m budou vylou&#269;eni z p&#345;edm&#283;tu.</strong></li>
<li><strong>Je zak&#225;z&#225;no s k&#253;mkoliv sd&#237;let nebo mu ukazovat sv&#233; &#345;e&#353;en&#237; &#250;lohy.</strong></li>
</ul>
<h2 id="u&#382;ite&#269;n&#233;-funkce">U&#382;ite&#269;n&#233; funkce</h2>
<ul>
<li>
<a href="https://devdocs.io/c/io/fgets"><code>fgets</code></a> - na&#269;ten&#237; &#345;&#225;dku ze vstupu do &#345;et&#283;zce (pole znak&#367;).</li>
<li>
<a href="https://devdocs.io/c/string/byte/atoi"><code>atoi</code></a> - p&#345;eveden&#237; &#345;et&#283;zce obsahuj&#237;c&#237;ho &#269;&#237;slice na &#269;&#237;slo (<code>int</code>).</li>
<li>
<a href="https://devdocs.io/c/string/byte/strlen"><code>strlen</code></a> - zji&#353;t&#283;n&#237; d&#233;lky &#345;et&#283;zce.</li>
<li>
<a href="https://devdocs.io/c/string/byte/strtok"><code>strtok</code></a> - rozd&#283;len&#237; &#345;et&#283;zce nap&#345;. podle mezer.</li>
</ul>
<h2 id="testov&#225;n&#237;-programu">Testov&#225;n&#237; programu</h2>
<p>Uk&#225;zkov&#233; vstupy a v&#253;stupy naleznete v z&#225;lo&#382;ce <a href="#tests"><code>Tests</code></a>. Odtud si je tak&#233; m&#367;&#382;ete st&#225;hnout. Po nahr&#225;n&#237; zdrojov&#233;ho souboru se m&#367;&#382;ete pod&#237;vat, jestli testy pro&#353;ly nebo ne. <strong>Testujte ale prim&#225;rn&#283; sv&#233; &#345;e&#353;en&#237; na lok&#225;ln&#237;m po&#269;&#237;ta&#269;i.</strong> Takto m&#367;&#382;ete nap&#345;&#237;klad spustit v&#225;&#353; program se vstupn&#237;m souborem a zkontrolovat, &#382;e vydal spr&#225;vn&#253; v&#253;stup:</p>
<pre class="bash"><code>$ ./main 1 2 &lt; test_XX/stdin &gt; output
$ diff test_XX/stdout output</code></pre>
<p><strong>Nezapome&#328;te p&#345;edat spr&#225;vn&#233; parametry p&#345;&#237;kazov&#233; &#345;&#225;dky podle vzoru v testu!</strong></p>
<p>M&#367;&#382;ete tak&#233; pou&#382;&#237;t pomocn&#253; Python skript, kter&#253; si m&#367;&#382;ete st&#225;hnout (spolu se v&#353;emi testy) pomoc&#237; tla&#269;&#237;tka <code>Download all tests</code> v z&#225;lo&#382;ce <a href="#tests"><code>Tests</code></a>. Skript m&#367;&#382;ete takto:</p>
<pre class="bash"><code>$ python3 run-tests.py &lt;cesta-k-bin&#225;rn&#237;mu-souboru-va&#353;eho-&#345;e&#353;eni&gt;</code></pre>
<p>To, &#382;e v&#353;echny testy pro&#353;ly, v&#353;ak je&#353;t&#283; neznamen&#225;, &#382;e je v&#225;&#353; program spr&#225;vn&#283; :) Z&#225;rove&#328; pokud n&#283;kter&#233; testy neprojdou, neznamen&#225; to je&#353;t&#283; samo o sob&#283;, &#382;e va&#353;e &#345;e&#353;en&#237; dostane <code>0</code> bod&#367;.</p>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
</div>