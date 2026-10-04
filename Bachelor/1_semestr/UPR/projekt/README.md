<div>
<p>Term&#237;ny, bodov&#225;n&#237; a dal&#353;&#237; informace o projektech naleznete <a href="https://github.com/geordi/upr-course/blob/master/faq/projects.md">zde</a>. Vyberte si jedno z n&#225;sleduj&#237;c&#237;ch dvou zad&#225;n&#237; projektu (klikn&#283;te na zad&#225;n&#237; pro jeho zobrazen&#237;). Do <strong>31. 10. 2023</strong> mus&#237;te sv&#233;mu cvi&#269;&#237;c&#237;mu nahl&#225;sit, kter&#233; ze dvou zad&#225;n&#237; jste si zvolili!</p>
<div class="projects">


<div class="project">
<h1 class="title" title="Zobrazit/skr&#253;t zad&#225;n&#237;">
League of Legends statistiky
</h1>
<div class="content content-hidden">
<p>Napi&#353;te program, kter&#253; bude po&#269;&#237;tat statistiky hr&#225;&#269;&#367; hry <a href="https://leagueoflegends.com">League of Legends</a> (LoL) podle z&#225;znam&#367; LoL her. Tyto statistiky budou vyps&#225;ny ve form&#283; n&#283;jak&#233;ho grafick&#233;ho v&#253;stupu (HTML str&#225;nka, SVG graf, &#8230;).</p>
<h2 id="vstup-programu">Vstup programu</h2>
<p>Program bude povinn&#283; akceptovat t&#345;i argumenty z p&#345;&#237;kazov&#233; &#345;&#225;dky. Argumenty budou v&#382;dy zad&#225;ny v n&#225;sleduj&#237;c&#237;m po&#345;ad&#237;:</p>
<ul>
<li>Cesta k souboru se z&#225;znamy LoL her.</li>
<li>Cesta k souboru s p&#345;ezd&#237;vkami hr&#225;&#269;&#367;.</li>
<li>Cesta k v&#253;stupn&#237;mu souboru, do kter&#233;ho budou zaps&#225;ny statistiky vypo&#269;&#237;tan&#233; programem.</li>
</ul>
<p>Pokud program neobdr&#382;&#237; dostatek vstupn&#237;ch argument&#367;, tak vyp&#237;&#353;e chybovou hl&#225;&#353;ku a ukon&#269;&#237; se s chybov&#253;m k&#243;dem <code>1</code>.</p>
<p><em>Pozn&#225;mka: pokud budete z&#225;pasit s na&#269;&#237;t&#225;n&#237;m vstup&#367; p&#345;&#237;kazov&#233;ho &#345;&#225;dku, tak si klidn&#283; nejprve vstupy programu zadejte &#8220;natvrdo&#8221; ve funkci <code>main</code>, a k na&#269;&#237;t&#225;n&#237; vstup&#367; z p&#345;&#237;kazov&#233; &#345;&#225;dky se vra&#357;te pozd&#283;ji, a&#357; se na tom zbyte&#269;n&#283; nezaseknete.</em></p>
<h2 id="na&#269;ten&#237;-dat">Na&#269;ten&#237; dat</h2>
<p>Nejprve na&#269;t&#283;te obsah souboru se z&#225;znamy her. V souboru bude n&#283;kolik z&#225;znam&#367; o jednotliv&#253;ch LoL hr&#225;ch, kter&#233; prob&#283;hly po sob&#283; ve stejn&#233;m po&#345;ad&#237;, v jak&#233;m jsou v souboru. Ka&#382;d&#253; z&#225;znam jedn&#233; hry bude m&#237;t n&#225;sleduj&#237;c&#237; strukturu:</p>
<ul>
<li>Na prvn&#237;m &#345;&#225;dku z&#225;znamu mus&#237; b&#253;t &#345;et&#283;zec <code>match</code>.</li>
<li>Na druh&#233;m &#345;&#225;dku budou ID hr&#225;&#269;&#367;, kte&#345;&#237; hr&#225;li za &#269;erven&#253; t&#253;m (po&#269;et hr&#225;&#269;&#367; mus&#237; b&#253;t <code>3</code>).
<ul>
<li>Jednotliv&#225; ID budou odd&#283;len&#225; &#269;&#225;rkou.</li>
</ul>
</li>
<li>Na dal&#353;&#237;m &#345;&#225;dku budou postupn&#283; hodnoty zabit&#237; (K), asistenc&#237; (A) a smrt&#237; (D) pro ka&#382;d&#233;ho hr&#225;&#269;e.
<ul>
<li>Tyto t&#345;i hodnoty budou odd&#283;leny st&#345;edn&#237;kem, hodnoty pro jednotliv&#233; hr&#225;&#269;e budou odd&#283;leny &#269;&#225;rkou.</li>
</ul>
</li>
<li>Na dal&#353;&#237;ch dvou &#345;&#225;dc&#237;ch budou obdobn&#233; informace pro modr&#253; t&#253;m.</li>
<li>Na posledn&#237;m &#345;&#225;dku z&#225;znamu bude v&#253;sledek z&#225;znamu (hry).
<ul>
<li>Pokud zde bude &#345;et&#283;zec <code>red</code>, tak vyhr&#225;l &#269;erven&#253; t&#253;m.</li>
<li>Pokud zde bude &#345;et&#283;zec <code>blue</code>, tak vyhr&#225;l modr&#253; t&#253;m.</li>
</ul>
</li>
</ul>
<p>Uk&#225;zkov&#253; obsah souboru se z&#225;znamy:</p>
<pre><code>match
1,2,3
10;3;2,8;2;0,2;5;4
8,4,9
1;4;10,5;1;2,0;2;8
red
match
2,1,8
5;2;4,8;4;4,4;4;3
13,22,10
4;4;10,5;1;4,2;2;3
blue</code></pre>
<p>D&#225;le na&#269;t&#283;te soubor s p&#345;ezd&#237;vkami hr&#225;&#269;&#367;. Na ka&#382;d&#233;m &#345;&#225;dku souboru bude dvojice odd&#283;len&#225; &#269;&#225;rkou. Prvn&#237;m &#269;lenem dvojice bude ID hr&#225;&#269;e a druh&#253;m &#269;lenem jeho p&#345;ezd&#237;vka.</p>
<p>Uk&#225;zkov&#253; obsah souboru s n&#225;zvy hr&#225;&#269;&#367;:</p>
<pre><code>1,darksider
2,yasuo4life
4,heimerdanger</code></pre>
<p>Obsah souboru pou&#382;ijte pro sp&#225;rov&#225;n&#237; ID hr&#225;&#269;&#367; s jejich p&#345;ezd&#237;vkou p&#345;i v&#253;pisu statistiky.</p>
<p>Pokud vstupn&#237; data v jak&#233;mkoliv ze vstupn&#237;ch soubor&#367; nebudou odpov&#237;dat zadan&#233;mu form&#225;tu, vypi&#353;te chybu a ukon&#269;ete program. Stejn&#283; tak zkontrolujte, jestli vstupn&#237; data d&#225;vaj&#237; smysl (i kdyby byla ve spr&#225;vn&#233;m form&#225;tu). Nap&#345;&#237;klad ned&#225;v&#225; smysl, aby v r&#225;mci jednoho z&#225;pasu hr&#225;l hr&#225;&#269; se stejn&#253;m ID za v&#237;ce LoL postav. Chybou je d&#225;le nap&#345;&#237;klad i to, pokud by ID hr&#225;&#269;e v z&#225;znamu hry chyb&#283;lo v seznamu p&#345;ezd&#237;vek.</p>
<details>
<p><summary>P&#345;&#237;klady nevalidn&#237;ch vstup&#367;</summary></p>
<pre><code>match
1,2,3
10;3;2,8;2;0,2;5;4
8,4,9</code></pre>
<pre><code>match
1,3,3
10;3;2,8;2;0,2;5;4
8,4,9
1;4;10,5;1;2,0;2;8
red</code></pre>
<pre><code>match
1,2,3
10;3;2,8;2;0,2;5;4
8,4,3
1;4;10,5;1;2,0;2;8
red</code></pre>
<pre><code>match
1,2,3
10;3;2,8;2;0,2;5;4
red</code></pre>
<pre><code>match
1,2,3
10;3;2,8;2;0,2;5;4
8,4,9
1;4;10,5;1;2,0;2;8
redx</code></pre>
<pre><code>match
1,2,3
10;2,8;2;0,2;5;4
8,4,9
1;4;10,5;1;2,0;2;8
red</code></pre>
</details>
<p>Vstupn&#237; soubory pro v&#225;&#353; program si vytvo&#345;te ru&#269;n&#283;, m&#367;&#382;ete tak&#233; vyu&#382;&#237;t uk&#225;zkov&#233; vstupy v&#253;&#353;e.</p>
<h2 id="v&#253;po&#269;et-statistik">V&#253;po&#269;et statistik</h2>
<p>Jakmile na&#269;tete informace o jednotliv&#253;ch z&#225;pasech a jeho hr&#225;&#269;&#237;ch, vypo&#269;t&#283;te n&#283;jak&#233; zaj&#237;mav&#233; statistiky, a vypi&#353;te je nap&#345;. ve form&#283; HTML str&#225;nky do v&#253;stupn&#237;ho souboru (viz n&#237;&#382;e).</p>
<p>Ve v&#253;stupu se mus&#237; objevit z&#225;kladn&#237; statistika (nap&#345;. ve form&#283; tabulky) pro ka&#382;d&#233;ho hr&#225;&#269;e:</p>
<ul>
<li>Kolik m&#225; kill&#367;, asistenc&#237; a smrt&#237;</li>
<li>Kolik z&#225;pas&#367; hr&#225;l</li>
<li>Kolikr&#225;t vyhr&#225;l a kolikr&#225;t prohr&#225;l</li>
<li>Kolikr&#225;t hr&#225;l za dan&#233; barvy t&#253;mu</li>
</ul>
<p>D&#225;le spo&#269;&#237;tejte alespo&#328; t&#345;i navazuj&#237;c&#237; statistiky, kter&#233; si vymysl&#237;te. M&#367;&#382;e to b&#253;t nap&#345;.:</p>
<ul>
<li>
<a href="https://leagueoflegends.fandom.com/wiki/Kill_to_Death_Ratio">Pom&#283;r zabit&#237; a smrt&#237;</a> pro jednotliv&#233; hr&#225;&#269;e</li>
<li>Seznam N hr&#225;&#269;&#367; s nejv&#237;ce killy</li>
<li>Seznam N hr&#225;&#269;&#367; s nejv&#237;ce v&#253;hrami</li>
<li>Nej&#269;ast&#283;j&#353;&#237; spoluhr&#225;&#269; pro ka&#382;d&#233;ho hr&#225;&#269;e</li>
<li>SVG graf s celkov&#253;m po&#269;tem kill&#367; po jednotliv&#253;ch z&#225;pasech (osa X = po&#345;ad&#237; z&#225;pasu, osa Y = po&#269;et kill&#367;)</li>
</ul>
<p><strong>P&#345;i v&#253;pisu statistik hr&#225;&#269;&#367; nevypisujte jejich ID, ale jejich p&#345;ezd&#237;vku! Tu mus&#237;te zjistit ze souboru s p&#345;ezd&#237;vkami, podle ID dan&#233;ho hr&#225;&#269;e.</strong></p>
<details>
<p><summary>V&#253;po&#269;et ELO ratingu</summary> Pokud byste cht&#283;li sv&#233; &#345;e&#353;en&#237; vy&#353;perkovat, m&#367;&#382;ete pro jednotliv&#233; hr&#225;&#269;e zkusit vypo&#269;&#237;tat jejich ELO rating a z n&#283;j dopo&#269;&#237;tat jejich divizi. <strong>V&#253;po&#269;et ELO ratingu nen&#237; povinn&#253;.</strong></p>
<p>Vypo&#269;t&#283;te tzv. <a href="https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details">ELO rating</a> pro ka&#382;d&#233;ho hr&#225;&#269;e. Postupn&#283; projd&#283;te v&#353;echny z&#225;pasy jeden po druh&#233;m a aktualizujte hodnoty ratingu pro jednotliv&#233; hr&#225;&#269;e. Ka&#382;d&#233;mu hr&#225;&#269;i po ka&#382;d&#233;m jeho z&#225;pasu aktualizujte jeho rating pomoc&#237; n&#225;sleduj&#237;c&#237;ho vzorce:</p>
<pre><code>ra = /* rating hr&#225;&#269;e */
rb = /* rating soupe&#345;e, vypo&#269;&#237;tejte ho jako pr&#367;m&#283;rnou hodnotu ratingu soupe&#345;ova t&#253;mu */

ea = 1 / (1 + 10^((rb - ra) / 400))  // znak `^` zna&#269;&#237; mocninu.
k = 30
sa = /* 1 pokud hr&#225;&#269;&#367;v t&#253;m vyhr&#225;l, 0, pokud prohr&#225;l */

novy_rating = ra + k * (sa - ea)</code></pre>
<p>Poprv&#233;, kdy&#382; v seznamu z&#225;pas&#367; naraz&#237;te na n&#283;jak&#233;ho hr&#225;&#269;e, inicializujte jeho rating na hodnotu <code>1000</code> (jinak &#345;e&#269;eno, &#250;vodn&#237; hodnota ratingu ka&#382;d&#233;ho hr&#225;&#269;e je <code>1000</code>). P&#345;i zpracov&#225;n&#237; z&#225;pasu pou&#382;ijte ve&#353;ker&#233; hodnoty ratingu pro v&#353;echny hr&#225;&#269;e takov&#233;, jak&#233; byly p&#345;ed t&#237;mto z&#225;pasem.</p>
<p>Jakmile projdete v&#353;echny z&#225;pasy a vypo&#269;tete fin&#225;ln&#237; rating ka&#382;d&#233;ho hr&#225;&#269;e, za&#345;a&#271;te ka&#382;d&#233;ho hr&#225;&#269;e podle jeho ratingu do jedn&#233; z n&#225;sleduj&#237;c&#237;ch diviz&#237;:</p>
<ul>
<li>
<code>Bronze</code> - rating mezi <code>0</code> a <code>1149</code>
</li>
<li>
<code>Silver</code> - rating mezi <code>1150</code> a <code>1499</code>
</li>
<li>
<code>Gold</code> - rating mezi <code>1500</code> a <code>1849</code>
</li>
<li>
<code>Platinum</code> - rating mezi <code>1850</code> a <code>2199</code>
</li>
<li>
<code>Diamond</code> - rating <code>2200</code> a v&#253;&#353;e</li>
</ul>
</details>
<h2 id="z&#225;pis-v&#253;sledku">Z&#225;pis v&#253;sledku</h2>
<p>Vytvo&#345;te v&#253;stupn&#237; soubor, kter&#253; bude obsahovat vypo&#269;ten&#233; statistiky jednotliv&#253;ch hr&#225;&#269;&#367;. Form&#225;t v&#253;stupu je na v&#225;s, ale ide&#225;ln&#237; by bylo vyu&#382;&#237;t HTML str&#225;nku s tabulkami a <a href="https://css-tricks.com/how-to-make-charts-with-svg/">SVG grafem</a>. Na podob&#283; v&#253;stupu se dop&#345;edu domluvte se sv&#253;m cvi&#269;&#237;c&#237;m.</p>
<h2 id="p&#345;&#237;klad-pou&#382;it&#237;">P&#345;&#237;klad pou&#382;it&#237;</h2>
<pre class="bash"><code>$ ./lol-stats matches.txt players.txt output.html</code></pre>
<h2 id="pozn&#225;mky-k-&#345;e&#353;en&#237;">Pozn&#225;mky k &#345;e&#353;en&#237;</h2>
<ul>
<li><p>O&#353;et&#345;te v programu chyby, kter&#233; m&#367;&#382;ou nastat, nap&#345;. &#353;patn&#233; vstupn&#237; parametry. V p&#345;&#237;pad&#283; chyby vypi&#353;te chybovou hl&#225;&#353;ku a ukon&#269;ete program s chybov&#253;m k&#243;dem <code>1</code>.</p></li>
<li><p>Projekt rozumn&#283; rozd&#283;lte do v&#237;ce <code>.c</code>/<code>.h</code> soubor&#367; a vytvo&#345;te k n&#283;mu bash skript, <code>Makefile</code> nebo <code>CMakeLists.txt</code> soubor, aby &#353;el projekt p&#345;elo&#382;it i na jin&#233;m po&#269;&#237;ta&#269;i (viz <a href="https://mrlvsb.github.io/upr-skripta/c/automatizace_prekladu.html">automatizace p&#345;ekladu</a>).</p></li>
<li><p>Vytvo&#345;te jednoduch&#253; README soubor, kter&#253; bude popisovat, jak program funguje a jak ho p&#345;elo&#382;it.</p></li>
<li><p>P&#345;i v&#253;voji pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> a/nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>! Velmi v&#225;m to usnadn&#237; detekci pam&#283;&#357;ov&#253;ch chyb. Odevzdan&#253; program nesm&#237; p&#345;i pou&#382;it&#237; Address sanitizeru ani Valgrindu vyvolat &#382;&#225;dn&#233; pam&#283;&#357;ov&#233; chyby.</p></li>
</ul>
<h2 id="konzultace-a-odevzd&#225;v&#225;n&#237;">Konzultace a odevzd&#225;v&#225;n&#237;</h2>
<ul>
<li>P&#345;&#237;padn&#233; nejasnosti v zad&#225;n&#237; a Va&#353;e &#345;e&#353;en&#237; pr&#367;b&#283;&#382;n&#283; konzultujte se sv&#253;m cvi&#269;&#237;c&#237;m nebo s Jakubem Ber&#225;nkem na <a href="https://discord.com/channels/631124326522945546/1058362395896062042">Discordu</a> (Kobzol), abyste p&#345;ede&#353;li p&#345;&#237;padn&#253;m nedorozum&#283;n&#237;m v interpretaci v&#253;sledku.</li>
<li>&#344;e&#353;en&#237; pr&#367;b&#283;&#382;n&#283; nahr&#225;vejte do syst&#233;mu <a href="http://kelvin.cs.vsb.cz">Kelvin</a>.</li>
<li><strong>Do Kelvina nahr&#225;vejte pouze soubory, kter&#233; jsou pot&#345;ebn&#233; k p&#345;ekladu! Tedy zejm&#233;na hlavi&#269;kov&#233; a zdrojov&#233; soubory, skripty k p&#345;elo&#382;en&#237; programu, p&#345;&#237;padn&#283; n&#283;jak&#233; obr&#225;zky &#269;i zvuky u SDL her. Nenahr&#225;vejte do Kelvina bin&#225;rn&#237; soubory ani &#382;&#225;dn&#233; dal&#353;&#237; zbyte&#269;n&#233; soubory.</strong></li>
<li><strong>Na projektu pracujte samostatn&#283;. Pokud budete opisovat od spolu&#382;&#225;k&#367;, st&#225;hnete si hotov&#233; &#345;e&#353;en&#237; z internetu nebo si nech&#225;te projekt n&#283;k&#253;m vytvo&#345;it a nebudete ho schopni obh&#225;jit osobn&#283; s cvi&#269;&#237;c&#237;m, tak budete vylou&#269;eni z p&#345;edm&#283;tu.</strong></li>
</ul>
</div>
</div>
<hr>
<div class="project">
<h1 class="title" title="Zobrazit/skr&#253;t zad&#225;n&#237;">
Bul&#225;nci
</h1>
<div class="content content-hidden">
<p>Naprogramujte n&#225;sleduj&#237;c&#237; hru pomoc&#237; knihovny <a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/sdl.html">SDL</a>. Popsan&#233; vlastnosti a uk&#225;zka hry jsou orienta&#269;n&#237;, m&#367;&#382;ete si hru &#269;&#225;ste&#269;n&#283; upravit dle sebe. Zam&#253;&#353;lenou formu hry a jej&#237; vlastnosti ale <strong>mus&#237;te nejprve prokonzultovat se sv&#253;m cvi&#269;&#237;c&#237;m</strong>.</p>
<p><a href="https://cs.wikipedia.org/wiki/Bul%C3%A1nci">Bul&#225;nci</a></p>
<p>V t&#233;to h&#345;e hr&#225;&#269;i ovl&#225;daj&#237; postavu Bul&#225;nka (pol&#353;t&#225;&#345;e), se kterou se pohybuj&#237; po jednoduch&#233; map&#283;, st&#345;&#237;l&#237; z n&#283;kolika typ&#367; zbran&#237; a sna&#382;&#237; se zni&#269;it ostatn&#237; hr&#225;&#269;e (ovl&#225;dan&#233; bu&#271; &#269;lov&#283;kem nebo po&#269;&#237;ta&#269;em). C&#237;lem je v &#269;asov&#233;m limitu dos&#225;hnout co nejv&#283;t&#353;&#237;ho sk&#243;re za co nejv&#237;ce zabit&#237; jin&#253;ch postav.</p>
<ul>
<li>Na za&#269;&#225;tku hry um&#237;st&#283;te n&#283;kolik postav na n&#225;hodn&#233; pozice do hern&#237; mapy (sta&#269;&#237; st&#225;hnout z internetu n&#283;jak&#253; obr&#225;zek pozad&#237;).
<ul>
<li>Na map&#283; vytvo&#345;te m&#237;sta, p&#345;es kter&#225; nep&#367;jde proj&#237;t, ani p&#345;es n&#283; st&#345;&#237;let (ide&#225;ln&#283; tak, aby odpov&#237;dala zvolen&#233;mu pozad&#237;).</li>
<li>P&#345;ezd&#237;vku a typ ovl&#225;d&#225;n&#237; (&#269;lov&#283;k/po&#269;&#237;ta&#269;) ka&#382;d&#233;ho hr&#225;&#269;e m&#367;&#382;ete navolit v menu, nebo jej na&#269;&#237;st ze vstupn&#237;ho souboru.</li>
</ul>
</li>
<li>Ka&#382;d&#225; postava bude moct chodit p&#345;es pr&#367;choz&#237; m&#237;sta, ot&#225;&#269;et se nahoru/dol&#367;/doleva/doprava a st&#345;&#237;let.
<ul>
<li>Efekt st&#345;ely z&#225;vis&#237; na pr&#225;v&#283; aktivn&#237; zbrani (nap&#345;. pistole &#269;i brokovnice), viz origin&#225;ln&#237; hra.</li>
<li>Jakmile ve zbrani dojdou n&#225;boje, tak se postav&#283; vr&#225;t&#237; p&#367;vodn&#237; zbra&#328; (pistole), kter&#225; m&#225; nekone&#269;no n&#225;boj&#367;.</li>
</ul>
</li>
<li>Ka&#382;dou postavu m&#367;&#382;e ovl&#225;dat bu&#271; &#269;lov&#283;k nebo po&#269;&#237;ta&#269;.
<ul>
<li>Vytvo&#345;te n&#283;kolik ovl&#225;dac&#237;ch sch&#233;mat, aby mohlo v&#237;ce hr&#225;&#269;&#367; hr&#225;t na stejn&#233; kl&#225;vesnici (nap&#345;. WASD + R, &#353;ipky + M atd.).</li>
<li>Zkuste vytvo&#345;it jednoduchou logiku po&#269;&#237;ta&#269;em ovl&#225;dan&#233;ho hr&#225;&#269;e, kter&#253; se bude sna&#382;it chodit po map&#283; tak, aby se bl&#237;&#382;il k ostatn&#237;m hr&#225;&#269;&#367;m a sna&#382;il se je trefit.</li>
</ul>
</li>
<li>Pokud jeden hr&#225;&#269; zas&#225;hne jin&#233;ho hr&#225;&#269;e, tak se mu zv&#253;&#353;&#237; sk&#243;re.
<ul>
<li>Postava zasa&#382;en&#233;ho hr&#225;&#269;e p&#345;ehraje n&#283;jakou jednoduchou animaci &#250;mrt&#237;.</li>
<li>Zasa&#382;en&#253; hr&#225;&#269; se po n&#283;kolika vte&#345;in&#225;ch vr&#225;t&#237; zp&#283;t do hry na n&#225;hodn&#233; pozici na map&#283;.</li>
</ul>
</li>
<li>V n&#225;hodn&#253;ch intervalech se budou na map&#283; objevovat zbran&#283;, kter&#233; lze postavou sebrat a pot&#233; je pou&#382;&#237;vat. Po sebr&#225;n&#237; zbra&#328; z mapy zmiz&#237;.</li>
<li>Jakmile vypr&#353;&#237; &#269;asov&#253; limit, hra kon&#269;&#237;. Program pot&#233; vyp&#237;&#353;e tabulku se sk&#243;re jednotliv&#253;ch hr&#225;&#269;&#367;.</li>
</ul>
<p>Odd&#283;lte samotnou logiku hry od jej&#237;ho vykreslov&#225;n&#237;. M&#283;lo by tedy j&#237;t pou&#382;&#237;vat hru nez&#225;visle na SDL pomoc&#237; odpov&#237;daj&#237;c&#237;ch struktur a funkc&#237;. Logiku hry a vykreslov&#225;n&#237; tedy i um&#237;st&#283;te do jin&#253;ch zdrojov&#253;ch soubor&#367;.</p>
<h2 id="vzorov&#225;-uk&#225;zka-hry">Vzorov&#225; uk&#225;zka hry</h2>
<p>M&#367;&#382;ete se pod&#237;vat na gameplay hry na <a href="https://youtu.be/_pTwY7gRIIg?t=31">YouTube</a>.</p>
<h2 id="pozn&#225;mky-k-&#345;e&#353;en&#237;-1">Pozn&#225;mky k &#345;e&#353;en&#237;</h2>
<ul>
<li><p>Nab&#237;dn&#283;te ve h&#345;e hr&#225;&#269;ovi &#250;vodn&#237; menu, kde si vybere konfiguraci hry.</p></li>
<li><p>Projekt rozumn&#283; rozd&#283;lte do v&#237;ce <code>.c</code>/<code>.h</code> soubor&#367; a vytvo&#345;te k n&#283;mu bash skript, <code>Makefile</code> nebo <code>CMakeLists.txt</code> soubor, aby &#353;el projekt p&#345;elo&#382;it i na jin&#233;m po&#269;&#237;ta&#269;i (viz <a href="https://mrlvsb.github.io/upr-skripta/c/automatizace_prekladu.html">automatizace p&#345;ekladu</a>).</p></li>
<li><p>Vytvo&#345;te jednoduch&#253; README soubor, kter&#253; bude popisovat, jak program funguje a jak ho p&#345;elo&#382;it.</p></li>
<li><p>P&#345;i v&#253;voji pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> a/nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>! Velmi v&#225;m to usnadn&#237; detekci pam&#283;&#357;ov&#253;ch chyb. Odevzdan&#253; program nesm&#237; p&#345;i pou&#382;it&#237; Address sanitizeru ani Valgrindu vyvolat &#382;&#225;dn&#233; pam&#283;&#357;ov&#233; chyby.</p></li>
<li><p>Uchov&#225;vejte seznam nejlep&#353;&#237;ch sk&#243;re hr&#225;&#269;e v souboru a zobrazte ho p&#345;i spu&#353;t&#283;n&#237; hry.</p></li>
<li><p>Grafick&#233; a zvukov&#233; materi&#225;ly (obr&#225;zky, pozad&#237;, zvukov&#233; efekty, &#8230;) si m&#367;&#382;ete st&#225;hnout z internetu, p&#345;&#237;padn&#283; vytvo&#345;it n&#283;jak&#233; vlastn&#237; jednoduch&#233; grafick&#233; prvky. Hra nemus&#237; vypadat &#250;&#382;asn&#283;, ani nemus&#237; b&#253;t p&#345;esnou kopi&#237; origin&#225;lu, hlavn&#283; mus&#237; fungovat :)</p></li>
</ul>
<h2 id="konzultace-a-odevzd&#225;v&#225;n&#237;-1">Konzultace a odevzd&#225;v&#225;n&#237;</h2>
<ul>
<li>P&#345;&#237;padn&#233; nejasnosti v zad&#225;n&#237; a Va&#353;e &#345;e&#353;en&#237; pr&#367;b&#283;&#382;n&#283; konzultujte se sv&#253;m cvi&#269;&#237;c&#237;m nebo s Jakubem Ber&#225;nkem na <a href="https://discord.com/channels/631124326522945546/1058362395896062042">Discordu</a> (Kobzol), abyste p&#345;ede&#353;li p&#345;&#237;padn&#253;m nedorozum&#283;n&#237;m v interpretaci v&#253;sledku.</li>
<li>&#344;e&#353;en&#237; pr&#367;b&#283;&#382;n&#283; nahr&#225;vejte do syst&#233;mu <a href="http://kelvin.cs.vsb.cz">Kelvin</a>.</li>
<li><strong>Do Kelvina nahr&#225;vejte pouze soubory, kter&#233; jsou pot&#345;ebn&#233; k p&#345;ekladu! Tedy zejm&#233;na hlavi&#269;kov&#233; a zdrojov&#233; soubory, skripty k p&#345;elo&#382;en&#237; programu, p&#345;&#237;padn&#283; n&#283;jak&#233; obr&#225;zky &#269;i zvuky u SDL her. Nenahr&#225;vejte do Kelvina bin&#225;rn&#237; soubory ani &#382;&#225;dn&#233; dal&#353;&#237; zbyte&#269;n&#233; soubory.</strong></li>
<li><strong>Na projektu pracujte samostatn&#283;. Pokud budete opisovat od spolu&#382;&#225;k&#367;, st&#225;hnete si hotov&#233; &#345;e&#353;en&#237; z internetu nebo si nech&#225;te projekt n&#283;k&#253;m vytvo&#345;it a nebudete ho schopni obh&#225;jit osobn&#283; s cvi&#269;&#237;c&#237;m, tak budete vylou&#269;eni z p&#345;edm&#283;tu.</strong></li>
</ul>
</div>
</div>
<hr>
</div>
<p>Pokud byste v zad&#225;n&#237; &#269;emukoliv nerozum&#283;li nebo cht&#283;li s n&#283;&#269;&#237;m poradit, kontaktujte sv&#233;ho cvi&#269;&#237;c&#237;ho nebo Jakuba Ber&#225;nka (Discord: Kobzol, e-mail: jakub.beranek@vsb.cz).</p>