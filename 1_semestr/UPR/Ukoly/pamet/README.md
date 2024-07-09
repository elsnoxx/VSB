<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak funguje pam&#283;&#357; na pozad&#237; na&#353;ich program&#367;, jak sd&#237;let data (prom&#283;nn&#233;) mezi funkcemi a tak&#233; jak si vytvo&#345;it v&#237;ce prom&#283;nn&#253;ch najednou pomoc&#237; pol&#237;.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/c/prace_s_pameti/prace_s_pameti.html">Pr&#225;ce s pam&#283;t&#237;</a>
<ul>
<li>Kapitoly <a href="https://mrlvsb.github.io/upr-skripta/c/prace_s_pameti/automaticka_pamet.html">Automatick&#225; pam&#283;&#357;</a> a <a href="https://mrlvsb.github.io/upr-skripta/c/prace_s_pameti/ukazatele.html">Ukazatele</a>
</li>
</ul>
</li>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/c/pole/pole.html">Pole</a>
<ul>
<li>Kapitola <a href="https://mrlvsb.github.io/upr-skripta/c/pole/staticka_pole.html">Statick&#233; pole</a>
</li>
</ul>
</li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/ukazatele.html">zde</a> a <a href="https://mrlvsb.github.io/upr-skripta/ulohy/pole.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu.</strong></p>
<p>Tento t&#253;den si zkus&#237;te napsat program, kter&#253; bude po&#269;&#237;tat <a href="https://cs.wikipedia.org/wiki/Histogram">histogram</a> &#269;&#237;sel. Histogram bude m&#237;t <code>9</code> &#8220;ko&#353;&#367;&#8221; a bude zaznamen&#225;vat &#269;etnost jednotliv&#253;ch hodnot, kter&#233; budou v rozsahu zadan&#233;m na vstupu programu.</p>
<p>Program by se m&#283;l chovat takto:</p>
<ol type="1">
<li>Na&#269;te ze vstupu znak, kter&#253; bude ud&#225;vat, jestli se m&#225; vykreslit horizont&#225;ln&#237; (pokud na vstupu byl znak <code>h</code>) nebo vertik&#225;ln&#237; (pokud na vstupu byl znak <code>v</code>) histogram.
<ul>
<li>Pokud bude zad&#225;n jin&#253; znak ne&#382; <code>v</code> nebo <code>h</code>, vypi&#353;te &#345;&#225;dek <code>Neplatny mod vykresleni</code> na standardn&#237; v&#253;stup a ukon&#269;ete program s chybov&#253;m k&#243;dem <code>1</code> (viz <a href="https://mrlvsb.github.io/upr-skripta/ruzne/funkce_main.html">skripta</a>).</li>
</ul>
</li>
<li>Na&#269;te ze vstupu dv&#283; nez&#225;porn&#225; &#269;&#237;sla (<code>n</code> a <code>m</code>).
<ul>
<li>
<code>n</code> ud&#225;v&#225;, kolik program na&#269;te &#269;&#237;sel, ze kter&#253;ch bude po&#269;&#237;t&#225;n histogram.</li>
<li>
<code>m</code> ud&#225;v&#225; rozsah &#269;&#237;sel, pro kter&#233; m&#225;te po&#269;&#237;tat histogram. Tento rozsah bude d&#225;n intervalem <code>[m, m + 8]</code>.
<ul>
<li>Nap&#345;. kdy&#382; <code>m</code> bude <code>5</code>, tak histogram bude po&#269;&#237;tat v&#253;skyty &#269;&#237;sel <code>5</code> a&#382; <code>13</code> (v&#269;etn&#283;).</li>
</ul>
</li>
</ul>
</li>
<li>Na&#269;te od u&#382;ivatele <code>n</code> cel&#253;ch &#269;&#237;sel odd&#283;len&#253;ch mezerou a vypo&#269;&#237;t&#225; pro n&#283; histogram.
<ul>
<li>Pokud bude &#269;&#237;slo na vstupu mimo interval <code>[m, m + 8]</code>, pokl&#225;dejte takov&#233; &#269;&#237;slo za <em>nevalidn&#237;</em>. Po&#269;et nevalidn&#237;ch &#269;&#237;sel si v programu pamatujte.</li>
<li>Histogram po&#269;&#237;t&#225;, kolikr&#225;t se jednotliv&#233; hodnoty <code>m</code> a&#382; <code>m + 8</code> vyskytly ve vstupu.</li>
<li>Histogram reprezentujte polem.</li>
<li>Ide&#225;ln&#237; &#345;e&#353;en&#237; pro v&#253;po&#269;et histogramu nen&#237; m&#237;t pod sebou 9x <code>if</code> v&#283;tev. Zamyslete se, jak histogram spo&#269;&#237;tat bez podm&#237;nek, lze to prov&#233;st jedn&#237;m &#345;&#225;dkem.</li>
</ul>
</li>
<li>Pokud byl na vstupu znak <code>h</code>, vykresl&#237; horizont&#225;ln&#237; histogram na v&#253;stup.
<ul>
<li>Vykreslete pod sebe &#269;&#237;sla <code>m</code> a&#382; <code>m + 8</code> (ka&#382;d&#233; &#269;&#237;slo na jednom &#345;&#225;dku).
<ul>
<li>
<p>Zarovnejte &#269;&#237;sla doprava tak, aby po&#269;ty v&#253;skyt&#367; pro jednotliv&#225; &#269;&#237;sla (viz n&#237;&#382;e) za&#269;&#237;nala v&#382;dy ve stejn&#233;m sloupci. Nap&#345;&#237;klad, pokud bude <code>m=98</code>, tak <code>m + 8</code> bude <code>106</code>, a toto &#269;&#237;slo m&#225; v&#237;ce &#269;&#237;slic, ne&#382; <code>98</code>. T&#237;m p&#225;dem mus&#237;te p&#345;ed p&#345;i v&#253;pisu p&#345;ed <code>98</code> a <code>99</code> vypsat jednu mezeru, jinak by &#269;&#237;sla pod sebou nepasovala spr&#225;vn&#283;. Zde je uk&#225;zka (znak <code>_</code> zde zn&#225;zor&#328;uje mezeru):</p>
<pre><code>_98 ##
_99 #
100
101 ###
...</code></pre>
</li>
<li><p>N&#225;pov&#283;da: zkuste pop&#345;em&#253;&#353;let, jak m&#367;&#382;ete spo&#269;&#237;tat, kolik m&#225; &#269;&#237;slo &#269;&#237;slic, a tedy i kolik zabere znak&#367;.</p></li>
</ul>
</li>
<li>Pokud se dan&#233; &#269;&#237;slo vyskytlo na vstupu, tak za n&#283;j vykreslete mezeru a znak <code>#</code> tolikr&#225;t, kolikr&#225;t se dan&#233; &#269;&#237;slo vyskytlo na vstupu.</li>
<li>Pokud se na vstupu vyskytla n&#283;jak&#225; <em>nevalidn&#237;</em> &#269;&#237;sla, tak za posledn&#237;m &#345;&#225;dkem vypsan&#233;ho histogramu vypi&#353;te dal&#353;&#237; &#345;&#225;dek s textem <code>invalid:</code>, a za n&#237;m vypi&#353;te znak <code>#</code> tolikr&#225;t, kolik bylo nevalidn&#237;ch &#269;&#237;sel na vstupu.</li>
</ul>
</li>
<li>
<em>Bonus</em>: Pokud byl na vstupu znak <code>v</code>, vykresl&#237; vertik&#225;ln&#237; histogram na v&#253;stup.
<ul>
<li>Vykreslete &#269;&#237;sla <code>m</code> a&#382; <code>m + 8</code> vedle sebe.</li>
<li>Pokud se dan&#233; &#269;&#237;slo vyskytlo na vstupu, tak nad n&#237;m by m&#283;l b&#253;t vykreslen&#253; sloupe&#269;ek se znakem <code>#</code>. Sloupe&#269;ek bude tak vysok&#253;, kolikr&#225;t se dan&#253; znak vyskytl na vstupu.</li>
<li>U vertik&#225;ln&#237;ho histogramu m&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e rozsah &#269;&#237;sel histogramu bude v&#382;dy <code>[1, 9]</code>.
<ul>
<li>Nevalidn&#237; &#269;&#237;sla vypi&#353;te v prvn&#237;m sloupci, a tento sloupec ozna&#269;te znakem <code>i</code>.</li>
</ul>
</li>
<li>Vertik&#225;ln&#237; histogram je bonus, proto&#382;e je slo&#382;it&#283;j&#353;&#237; na implementaci, ne&#382; horizont&#225;ln&#237;.</li>
<li>Pokud nebudete bonus implementovat, tak prost&#283; p&#345;edpokl&#225;dejte, &#382;e na za&#269;&#225;tku vstupu programu bude v&#382;dy <code>h</code>. Znak ale mus&#237;te st&#225;le na&#269;&#237;st, aby fungovaly testy v Kelvinu.</li>
</ul>
</li>
</ol>
<p>Pod&#237;vejte se na z&#225;lo&#382;ku <code>Tests</code> pro vzorov&#233; uk&#225;zky vstup&#367; a v&#253;stup&#367;.</p>
<p>Pozn&#225;mky ke k&#243;du:</p>
<ul>
<li>
<p><strong>Pro tuto &#250;lohu nen&#237; pot&#345;eba pou&#382;&#237;vat tzv. dynamickou alokaci pam&#283;ti. V&#353;e vy&#345;e&#353;te pouze pomoc&#237; ukazatel&#367; a statick&#253;ch pol&#237;. Nevyu&#382;&#237;vejte ani tzv. VLA (variable-length array):</strong></p>
<pre class="c"><code>  int velikost = ...;

  // Pole na z&#225;sobn&#237;ku, jeho&#382; velikost nen&#237; konstantn&#237; (VLA).
  // Nepou&#382;&#237;vejte tuto vlastnost C99.
  // https://stackoverflow.com/questions/12407754/what-technical-disadvantages-do-c99-style-vlas-have
  int pole[velikost];</code></pre>
</li>
<li><p>Zamyslete se nad t&#237;m, kde si uchov&#225;vat po&#269;et nevalidn&#237;ch hodnot (hodnot mimo zadan&#253; rozsah). Pot&#345;ebujete na to separ&#225;tn&#237; m&#237;sto v pam&#283;ti, nebo m&#367;&#382;ete vyu&#382;&#237;t histogram? :)</p></li>
<li><p>V zad&#225;n&#237; pou&#382;&#237;v&#225;me zkr&#225;cen&#233; n&#225;zvy pro ur&#269;it&#233; prom&#283;nn&#233; (nap&#345;. <code>n</code>). V programu si prom&#283;nn&#233; pojmenujte tak, aby bylo jasn&#233;, k &#269;emu slou&#382;&#237;, n&#225;zvy prom&#283;nn&#253;ch nemus&#237; p&#345;esn&#283; odpov&#237;dat zad&#225;n&#237;.</p></li>
<li><p>Prom&#283;nn&#233; pojmenov&#225;vejte tak, aby &#353;lo z jejich n&#225;zvu okam&#382;it&#283; poznat, k &#269;emu slou&#382;&#237;, a vytv&#225;&#345;ejte je co nejbl&#237;&#382;e m&#237;stu, kde jsou v k&#243;du opravdu pot&#345;eba (ne v&#353;echny na za&#269;&#225;tku funkce).</p></li>
<li><p>Rozd&#283;lte si program do funkc&#237; tak, aby to d&#225;valo smysl (nap&#345;. funkce na na&#269;ten&#237; histogramu, vykreslen&#237; histogramu).</p></li>
</ul>
<p>P&#345;&#237;klad vstupu:</p>
<pre><code>h
10 1
3 3 2 3 7 1 10 4 9 9</code></pre>
<ul>
<li>Hodnota <code>10</code> ud&#225;v&#225;, &#382;e na vstupu bude deset &#269;&#237;sel.</li>
<li>Hodnota <code>1</code> ud&#225;v&#225;, &#382;e histogram bude po&#269;&#237;tat v&#253;skyty &#269;&#237;sel <code>1</code> a&#382; <code>9</code>.</li>
</ul>
<p>Odpov&#237;daj&#237;c&#237; v&#253;stup:</p>
<pre><code>1 #
2 #
3 ###
4 #
5
6
7 #
8
9 ##
invalid: #</code></pre>
<h3 id="na&#269;&#237;t&#225;n&#237;-vstupu">Na&#269;&#237;t&#225;n&#237; vstupu</h3>
<p>Pomoc&#237; funkce <a href="https://devdocs.io/c/io/fscanf"><code>scanf</code></a> m&#367;&#382;ete na&#269;&#237;st cel&#233; &#269;&#237;slo ze vstupu programu do &#269;&#237;seln&#233; prom&#283;nn&#233; takto:</p>
<pre class="c"><code>int number;
scanf("%d", &amp;number);</code></pre>
<p><strong>Nemus&#237;te nijak &#345;e&#353;it b&#237;l&#233; znaky ani rozd&#283;lov&#225;n&#237; podle mezer/&#345;&#225;dk&#367;. Sta&#269;&#237; opakovan&#283; volat funkci <code>scanf</code>, o rozd&#283;len&#237; podle mezer se postar&#225; sama.</strong></p>
<p>Znak pot&#233; m&#367;&#382;ete na&#269;&#237;st takto:</p>
<pre class="c"><code>char ch;
scanf("%c", &amp;ch);</code></pre>
<h2 id="testov&#225;n&#237;-programu">Testov&#225;n&#237; programu</h2>
<p>Uk&#225;zkov&#233; vstupy a v&#253;stupy naleznete v z&#225;lo&#382;ce <code>Tests</code>. Odtud si je tak&#233; m&#367;&#382;ete st&#225;hnout (<code>stdin</code> - vstup, <code>stdout</code> - o&#269;ek&#225;v&#225;n&#253; v&#253;stup z va&#353;eho programu). Po nahr&#225;n&#237; zdrojov&#233;ho souboru se m&#367;&#382;ete pod&#237;vat, jestli testy pro&#353;ly nebo ne. To, &#382;e v&#353;echny testy pro&#353;ly, v&#353;ak je&#353;t&#283; neznamen&#225;, &#382;e je v&#225;&#353; program spr&#225;vn&#283; :) Stejn&#283; tak naopak, pokud v&#353;echny testy nepro&#353;ly, neznamen&#225; to automaticky, &#382;e m&#225;te nula bod&#367;.</p>
<h3 id="p&#345;esm&#283;rov&#225;n&#237;-vstupu">P&#345;esm&#283;rov&#225;n&#237; vstupu</h3>
<p>Abyste nemuseli neust&#225;le ru&#269;n&#283; zad&#225;vat &#269;&#237;sla z kl&#225;vesnice p&#345;i testov&#225;n&#237; programu, m&#367;&#382;ete data na vstup programu <a href="https://www.pslib.cz/milan.kerslager/BASH:_P%C5%99esm%C4%9Brov%C3%A1n%C3%AD">p&#345;esm&#283;rovat</a> ze souboru:</p>

# spu&#353;t&#283;n&#237; souboru, p&#345;esm&#283;rov&#225;n&#237; souboru 01.stdin na vstup programu
<pre class="sh"><code># p&#345;eklad programu
gcc -g -fsanitize=address main.c -o program


./program &lt; 01.stdin</code></pre>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
</div>