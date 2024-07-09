<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak m&#367;&#382;eme manu&#225;ln&#283; alokovat a dealokovat dynamickou pam&#283;&#357; na hald&#283; a k &#269;emu to m&#367;&#382;e b&#253;t u&#382;ite&#269;n&#233;. Uk&#225;&#382;eme si tak&#233;, jak vytv&#225;&#345;et v&#237;cerozm&#283;rn&#233; pole, kter&#233; se hod&#237; nap&#345;. pro reprezentaci obr&#225;zk&#367;.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/prace_s_pameti/dynamicka_pamet.html">Dynamick&#225; pam&#283;&#357;</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/pole/dynamicka_pole.html">Dynamick&#225; pole</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/pole/vicerozmerna_pole.html">V&#237;cerozm&#283;rn&#225; pole</a></li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/dvourozmerne_pole.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu.</strong></p>
<p>Tento t&#253;den si zkus&#237;te napsat program, kter&#253; vytvo&#345;&#237; dvourozm&#283;rn&#233; pl&#225;tno/m&#345;&#237;&#382;ku/obr&#225;zek, na kter&#233;m se budou pohybovat a kreslit &#382;elvi&#269;ky. U&#382;ivatel aplikace bude p&#345;&#237;kazy ovl&#225;dat, jak se budou &#382;elvi&#269;ky pohybovat, a bude jim tak&#233; d&#225;vat povel, kdy maj&#237; na pl&#225;tno kreslit &#8220;pixely&#8221;.</p>
<p>Program by se m&#283;l chovat takto:</p>
<ol type="1">
<li>Na&#269;t&#283;te ze vstupu dv&#283; &#269;&#237;sla odd&#283;len&#225; mezerou, po&#269;et &#345;&#225;dk&#367; pl&#225;tna (<code>rows</code>) a po&#269;et sloupc&#367; pl&#225;tna (<code>cols</code>).</li>
<li>Vytvo&#345;te <a href="https://mrlvsb.github.io/upr-skripta/c/pole/vicerozmerna_pole.html">2D pl&#225;tno</a> (pole) znak&#367; o rozm&#283;rech <code>rows</code> &#345;&#225;dk&#367; a <code>cols</code> sloupc&#367;.
<ul>
<li>Znaky ve m&#345;&#237;&#382;ce inicializujte na znak <code>'.'</code> (te&#269;ka).</li>
</ul>
</li>
<li>Vytvo&#345;te &#382;elvi&#269;ku na pozici <code>(0, 0)</code> - nult&#253; &#345;&#225;dek a nult&#253; sloupec (vlevo naho&#345;e).
<ul>
<li>&#381;elvi&#269;ka na za&#269;&#225;tku bude sm&#283;&#345;ovat doprava.</li>
<li>Ka&#382;d&#225; &#382;elvi&#269;ka bude m&#237;t svou pozici (&#345;&#225;dek, sloupec) a sm&#283;r, na kter&#253; je zrovna nato&#269;en&#225; (doprava, doleva, nahoru, dol&#367;).</li>
<li>Prvn&#237; sou&#345;adnice ur&#269;uje &#345;&#225;dek, druh&#225; sloupec (&#345;&#225;dky rostou dol&#367;, sloupce rostou doprava).</li>
</ul>
</li>
<li>
<a href="#na%C4%8D%C3%ADt%C3%A1n%C3%AD-vstupu">Na&#269;&#237;tejte</a> postupn&#283; znaky ze vstupu programu. Znaky budou odd&#283;len&#233; b&#237;l&#253;m znakem (mezerou &#269;i od&#345;&#225;dkov&#225;n&#237;m). Podle na&#269;ten&#233;ho znaku prove&#271;te n&#225;sleduj&#237;c&#237; akci:
<ul>
<li>Pokud naraz&#237;te na znak <code>r</code>, <a href="#ot%C3%A1%C4%8Den%C3%AD-%C5%BEelvi%C4%8Dek">oto&#269;te</a> sm&#283;r v&#353;ech &#382;elvi&#269;ek doprava.</li>
<li>Pokud naraz&#237;te na znak <code>l</code>, <a href="#ot%C3%A1%C4%8Den%C3%AD-%C5%BEelvi%C4%8Dek">oto&#269;te</a> sm&#283;r v&#353;ech &#382;elvi&#269;ek doleva.</li>
<li>Pokud naraz&#237;te na znak <code>m</code>, posu&#328;te v&#353;echy &#382;elvi&#269;ky o jednu pozici v jejich sm&#283;ru.
<ul>
<li><p>Pokud bude m&#237;t &#382;elvi&#269;ka sm&#283;r doprava, posune se z pozice <code>(r, s)</code> na <code>(r, s + 1)</code>.</p></li>
<li><p>Pokud bude m&#237;t &#382;elvi&#269;ka sm&#283;r doleva, posune se z pozice <code>(r, s)</code> na <code>(r, s - 1)</code>.</p></li>
<li><p>Pokud bude m&#237;t &#382;elvi&#269;ka sm&#283;r nahoru, posune se z pozice <code>(r, s)</code> na <code>(r - 1, s)</code>.</p></li>
<li><p>Pokud bude m&#237;t &#382;elvi&#269;ka sm&#283;r dol&#367;, posune se z pozice <code>(r, s)</code> na <code>(r + 1, s)</code>.</p></li>
<li>
<p>Pokud by jak&#225;koliv &#382;elvi&#269;ka m&#283;la prov&#233;st posun, kter&#253; by ji vyvedl ven z pl&#225;tna (nap&#345;. na pozici <code>(0, 0)</code> by se posunula doleva), tak &#382;elvi&#269;ku &#8220;teleportujte&#8221; na opa&#269;nou stranu m&#345;&#237;&#382;ky. Nap&#345;.:</p>
<pre><code>z..
...
...</code></pre>
<p>V <code>3x3</code> m&#345;&#237;&#382;ce v&#253;&#353;e je pozice &#382;elvi&#269;ky <code>(0, 0)</code> zn&#225;zorn&#283;na znakem <code>z</code>.</p>
<ul>
<li>Pokud by zde &#382;elvi&#269;ka sm&#283;&#345;ovala sm&#283;rem doleva, a dostala p&#345;&#237;kaz <code>m</code>, tak se p&#345;esune na pozici <code>(0, 2)</code>, tj. &#250;pln&#283; doprava nahoru.</li>
<li>Pokud by zde &#382;elvi&#269;ka sm&#283;&#345;ovala sm&#283;rem nahoru, a dostala p&#345;&#237;kaz <code>m</code>, tak se p&#345;esune na pozici <code>(2, 0)</code>, tj. &#250;pln&#283; doleva dol&#367;.</li>
</ul>
</li>
</ul>
</li>
<li>Pokud naraz&#237;te na znak <code>o</code>, zapi&#353;te na pl&#225;tno znak <code>o</code> na sou&#269;asn&#233; pozici ka&#382;d&#233; &#382;elvi&#269;ky.
<ul>
<li>Pokud ji&#382; na dan&#233;m m&#237;st&#283; byl znak <code>o</code>, tak m&#237;sto toho zapi&#353;te na dan&#233; m&#237;sto znak pr&#225;zdn&#233; pozice (<code>.</code>). Jinak &#345;e&#269;eno, kreslen&#237; na pozici, kde ji&#382; byl nakreslen pixel, dan&#253; pixel &#8220;vyma&#382;e&#8221;.</li>
<li>Pokud by do&#353;lo ke kreslen&#237; v moment&#283;, kdy na stejn&#233; pozici bude v&#237;ce &#382;elvi&#269;ek, tak by m&#283;lo doj&#237;t ke kreslen&#237; postupn&#283;. Tj. nejprve vykresl&#237; pixel prvn&#237; &#382;elvi&#269;ka, pot&#233; druh&#225; atd. To znamen&#225;, &#382;e pokud na stejn&#233;m m&#237;st&#283; budou dv&#283; &#382;elvy, tak se jejich vykreslen&#237; &#8220;vyru&#353;&#237;&#8221;. Pokud na stejn&#233;m m&#237;st&#283; budou t&#345;i &#382;elvy, tak se naopak pixel op&#283;t vykresl&#237;.</li>
</ul>
</li>
<li>Pokud naraz&#237;te na znak <code>f</code>, vytvo&#345;&#237; se nov&#253; paraleln&#237; vesm&#237;r a vznikne tak jedna nov&#225; &#382;elvi&#269;ka.
<ul>
<li>Nov&#283; vytvo&#345;en&#225; &#382;elvi&#269;ka bude um&#237;st&#283;na na pozici <code>(0, 0)</code> a bude sm&#283;&#345;ovat doprava.</li>
<li>Po proveden&#237; tohoto p&#345;&#237;kazu se v&#353;echny dal&#353;&#237; p&#345;&#237;kazy (<code>o</code>, <code>m</code>, <code>l</code>, <code>r</code>) budou vztahovat i na tuto nov&#283; vytvo&#345;enou &#382;elvi&#269;ku!</li>
<li>V&#353;echny &#382;elvi&#269;ky sd&#237;lej&#237; stejn&#233; pl&#225;tno.</li>
<li>Aby se nezhroutilo multiversum, paraleln&#237;ch vesm&#237;r&#367; nesm&#237; b&#253;t p&#345;&#237;li&#353; moc. Pokud ji&#382; existuj&#237; t&#345;i &#382;elvi&#269;ky, tak tento p&#345;&#237;kaz ignorujte. Po&#269;et &#382;elvi&#269;ek tedy v&#382;dy bude max. t&#345;i.</li>
</ul>
</li>
<li>Pokud naraz&#237;te na znak <code>x</code>, vypi&#353;te pl&#225;tno na v&#253;stup programu a ukon&#269;ete program.
<ul>
<li>M&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e na konci vstupu programu bude v&#382;dy znak <code>x</code>.</li>
</ul>
</li>
</ul>
</li>
</ol>
<p>P&#345;&#237;klad vstupu:</p>
<pre><code>3 3
o
m
r
m
o
m
l
m
o
x</code></pre>
<p>Odpov&#237;daj&#237;c&#237; v&#253;stup:</p>
<pre class="text"><code>o..
.o.
..o</code></pre>
<h3 id="ot&#225;&#269;en&#237;-&#382;elvi&#269;ek">Ot&#225;&#269;en&#237; &#382;elvi&#269;ek</h3>
<p>Ot&#225;&#269;en&#237; by m&#283;lo prob&#237;hat pomoc&#237; klasick&#233; rotace v 2D prostoru. Pokud se &#382;elvi&#269;ka bude d&#237;vat nap&#345;. doprava a dostanete p&#345;&#237;kaz <code>r</code>, tak se pot&#233; &#382;elvi&#269;ka bude d&#237;vat dol&#367;. Pokud se &#382;elvi&#269;ka bude d&#237;vat nahoru a dostanete p&#345;&#237;kaz <code>l</code>, tak se pot&#233; &#382;elvi&#269;ka bude d&#237;vat doleva. Zkuste si vytvo&#345;it funkce na rotace sm&#283;ru doleva/doprava. Zkuste se zamyslet, jestli se v&#225;m poda&#345;&#237; naleznout jednoduch&#253; vzorec pro tuto rotaci, abyste v k&#243;du nemuseli m&#237;t spoustu podm&#237;nek pro r&#367;zn&#233; p&#345;&#237;pady rotace.</p>
<h3 id="pozn&#225;mky-k-&#345;e&#353;en&#237;">Pozn&#225;mky k &#345;e&#353;en&#237;</h3>
<ul>
<li>Pro tuto &#250;lohu nevyu&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/c/pole/staticka_pole.html#konstantn%C3%AD-velikost-statick%C3%A9ho-pole">VLA</a> (variable-length array)! Tato &#250;loha je cvi&#269;en&#237;m na dynamickou alokaci.</li>
<li>M&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e vstup bude v&#382;dy validn&#237;, nemus&#237;te tedy &#345;e&#353;it &#382;&#225;dn&#233; dal&#353;&#237; mo&#382;n&#233; znaky na vstupu (krom&#283; b&#237;l&#253;ch znak&#367;, viz <a href="#na%C4%8D%C3%ADt%C3%A1n%C3%AD-vstupu">Na&#269;&#237;t&#225;n&#237; vstupu</a>).</li>
<li>Pro vytvo&#345;en&#237; m&#345;&#237;&#382;ky/pl&#225;tna vyu&#382;ijte dynamickou alokaci pam&#283;ti. Sta&#269;&#237; naalokovat jedno pole, nepou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/c/pole/zubata_pole.html">zubat&#225; pole</a>. Nezapome&#328;te pole uvolnit, jakmile u&#382; jej nebudete pot&#345;ebovat.</li>
<li>Na reprezentaci &#382;elvi&#269;ek naopak nepot&#345;ebujete dynamickou alokaci, proto&#382;e v&#382;dy budou max. t&#345;i &#382;elvi&#269;ky.</li>
<li>Abyste si tvorbu programu zjednodu&#353;ili, m&#367;&#382;ete si nejprve si &#250;lohu naprogramovat pouze s jednou &#382;elvi&#269;kou. Jakmile v&#225;m bude program fungovat, tak pot&#233; teprve p&#345;idejte podporu pro p&#345;&#237;kaz <code>f</code> a v&#237;ce &#382;elvi&#269;ek. Pokud se v&#225;m nepovede naimplementovat p&#345;&#237;kaz <code>f</code>, tak odevzdejte alespo&#328; &#345;e&#353;en&#237; s jednou &#382;elvi&#269;kou.</li>
<li>Pro zjednodu&#353;en&#237; lad&#283;n&#237; programu si m&#367;&#382;ete vytvo&#345;it funkci pro v&#253;pis m&#345;&#237;&#382;ky, kter&#225; z&#225;rove&#328; vyp&#237;&#353;e i &#382;elvi&#269;ky, a volat tuto funkci po ka&#382;d&#233;m na&#269;ten&#233;m znaku, abyste si mohli vizualizovat, kde zrovna &#382;elvi&#269;ky jsou a t&#345;eba i jak&#253;m sm&#283;rem jsou nato&#269;en&#233;. Zkuste tak&#233; pou&#382;&#237;vat <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#krokov%C3%A1n%C3%AD">krokov&#225;n&#237;</a>.</li>
<li>&#381;elvi&#269;ky nemaj&#237; b&#253;t na m&#345;&#237;&#382;ce vypsan&#233; na v&#253;stup programu nijak zn&#225;zorn&#283;ny, v m&#345;&#237;&#382;ce se budou objevovat pouze znaky <code>'o'</code>, kter&#233; &#382;elvi&#269;ky bude kreslit, p&#345;&#237;padn&#283; &#8220;pr&#225;zdn&#225; m&#237;sta&#8221; zn&#225;zorn&#283;n&#225; znakem <code>'.'</code>.</li>
<li>Pozici ani sm&#283;r &#382;elvi&#269;ek si neukl&#225;dejte v m&#345;&#237;&#382;ce. Ta slou&#382;&#237; pouze k pamatov&#225;n&#237; znak&#367;.</li>
</ul>
<p>D&#233;lka referen&#269;n&#237;ho &#345;e&#353;en&#237;: ~100 &#345;&#225;dk&#367;</p>
<h3 id="bonusov&#233;-&#250;koly">Bonusov&#233; &#250;koly</h3>
<ul>
<li>Vygenerujte na m&#345;&#237;&#382;ce po jej&#237;m vytvo&#345;en&#237; na n&#283;kolika n&#225;hodn&#253;ch m&#237;stech p&#345;ek&#225;&#382;ky (zn&#225;zorn&#283;n&#233; znakem <code>'#'</code>). Pokud &#382;elvi&#269;ka naraz&#237; na p&#345;ek&#225;&#382;ku, tak p&#345;es ni nebude moct p&#345;ej&#237;t. O generov&#225;n&#237; n&#225;hodn&#253;ch &#269;&#237;sel si m&#367;&#382;ete p&#345;e&#269;&#237;st <a href="https://mrlvsb.github.io/upr-skripta/ruzne/nahodna_cisla.html">zde</a>.</li>
<li>Vytvo&#345;te z programu animaci, viz zad&#225;n&#237; p&#345;edminul&#233; &#250;lohy. Po ka&#382;d&#233;m pohybu &#382;elvi&#269;ky nebo vykreslen&#237; pixelu vykreslete celou m&#345;&#237;&#382;ku. Pozici &#382;elvi&#269;ek zn&#225;zorn&#283;te znakem <code>z</code> (p&#345;&#237;padn&#283; m&#367;&#382;ete pro ka&#382;dou &#382;elvi&#269;ku pou&#382;&#237;t jin&#253; znak). Aby animace vypadala p&#283;kn&#283;, tak p&#345;ed vykreslen&#237;m m&#345;&#237;&#382;ky vy&#269;ist&#283;te obrazovku termin&#225;lu a um&#237;st&#283;te kurzor na po&#269;&#225;tek. M&#367;&#382;ete pro to pou&#382;&#237;t ANSI sekvenci <code>printf("\x1b[2J\x1b[1;1F");</code>. Aby animace v&#237;ce vynikla, pou&#382;ijte mezeru m&#237;sto te&#269;ky pro pr&#225;zdn&#233; m&#237;sto na m&#345;&#237;&#382;ce.</li>
</ul>
<p>P&#345;&#237;klady animac&#237; pro n&#283;kter&#233; testovac&#237; vstupy:</p>
<ul>
<li>
<p>Test <code>My favourite subject</code></p>
<p><img src="img/turtle-anim-upr.gif" width="300" height="150"></p>
</li>
<li>
<p>Test <code>Triple fill painters</code></p>
<p><img src="img/turtle-anim-triple-fill.gif" width="180" height="200"></p>
</li>
<li>
<p>Test <code>H in parallel</code></p>
<p><img src="img/turtle-anim-parallel-h.gif" width="180" height="200"></p>
</li>
</ul>
<p>Pokud budete implementovat bonusov&#233; &#250;koly, tak v&#225;m nebudou proch&#225;zet testy v Kelvinovi. Odevzd&#225;vejte v tom p&#345;&#237;pad&#283; &#345;e&#353;en&#237; v takov&#233; podob&#283;, aby testy pro&#353;ly, bonusov&#253; k&#243;d nechte zakomentovan&#253;, a v Kelvinu p&#345;idejte koment&#225;&#345; (tla&#269;&#237;tko <code>+</code> u &#269;&#237;sla n&#283;jak&#233;ho &#345;&#225;dku odevzdan&#233;ho k&#243;du, anebo v&#283;t&#353;&#237; tla&#269;&#237;tko <code>+</code> &#250;pln&#283; naho&#345;e v z&#225;lo&#382;ce <code>Source code</code>) s informac&#237;, &#382;e jste implementovali bonus.</p>
<h3 id="na&#269;&#237;t&#225;n&#237;-vstupu">Na&#269;&#237;t&#225;n&#237; vstupu</h3>
<p>Pomoc&#237; funkce <a href="https://devdocs.io/c/io/fscanf"><code>scanf</code></a> m&#367;&#382;ete na&#269;&#237;st &#269;&#237;slo ze vstupu programu do &#269;&#237;seln&#233; prom&#283;nn&#233; takto:</p>
<pre class="c"><code>int number;
scanf("%d", &amp;number);</code></pre>
<p>Znak m&#367;&#382;ete na&#269;&#237;st takto:</p>
<pre class="c"><code>char ch;
scanf("%c", &amp;ch);</code></pre>
<p>Pozn&#225;mka: <code>scanf("%c", &#8230;)</code> bude na&#269;&#237;tat i b&#237;l&#233; znaky (nap&#345;. od&#345;&#225;dkov&#225;n&#237;). Ty ignorujte.</p>
<h3 id="testov&#225;n&#237;-programu">Testov&#225;n&#237; programu</h3>
<p>Uk&#225;zkov&#233; vstupy a v&#253;stupy naleznete v z&#225;lo&#382;ce <code>Tests</code>. Odtud si je tak&#233; m&#367;&#382;ete st&#225;hnout (<code>stdin</code> - vstup, <code>stdout</code> - o&#269;ek&#225;v&#225;n&#253; v&#253;stup z va&#353;eho programu). Po nahr&#225;n&#237; zdrojov&#233;ho souboru se m&#367;&#382;ete pod&#237;vat, jestli testy pro&#353;ly nebo ne. To, &#382;e v&#353;echny testy pro&#353;ly, v&#353;ak je&#353;t&#283; neznamen&#225;, &#382;e je v&#225;&#353; program spr&#225;vn&#283; :) Stejn&#283; tak naopak, pokud v&#353;echny testy nepro&#353;ly, neznamen&#225; to automaticky, &#382;e m&#225;te nula bod&#367;.</p>
<h3 id="p&#345;esm&#283;rov&#225;n&#237;-vstupu">P&#345;esm&#283;rov&#225;n&#237; vstupu</h3>
<p>Abyste nemuseli neust&#225;le ru&#269;n&#283; zad&#225;vat &#269;&#237;sla z kl&#225;vesnice p&#345;i testov&#225;n&#237; programu, m&#367;&#382;ete data na vstup programu <a href="https://www.pslib.cz/milan.kerslager/BASH:_P%C5%99esm%C4%9Brov%C3%A1n%C3%AD">p&#345;esm&#283;rovat</a> ze souboru:</p>

# spu&#353;t&#283;n&#237; souboru, p&#345;esm&#283;rov&#225;n&#237; souboru stdin ve slo&#382;ce test-simple na vstup programu
<pre class="sh"><code># p&#345;eklad programu
gcc -g -fsanitize=address main.c -o program


./program &lt; test-simple/stdin</code></pre>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
</div>