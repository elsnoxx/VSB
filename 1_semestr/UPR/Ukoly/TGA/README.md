<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak m&#367;&#382;eme v jazyce <em>C</em> pracovat s obr&#225;zky pomoc&#237; obr&#225;zkov&#233;ho form&#225;tu <a href="https://en.wikipedia.org/wiki/Truevision_TGA">TGA</a>.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/soubory/soubory.html">Soubory</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/tga.html">TGA</a></li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/soubory.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny. Nenahr&#225;vejte do Kelvina &#382;&#225;dn&#233; <code>.tga</code> obr&#225;zky ani slo&#382;ku s fonty, Kelvin si pot&#345;ebn&#233; soubory pro testy poskytne s&#225;m.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu. Je zak&#225;z&#225;no sd&#237;let sv&#233; &#345;e&#353;en&#237; s ostatn&#237;mi, opisovat od ostatn&#237;ch, nechat si od ostatn&#237;ch diktovat k&#243;d a pou&#382;&#237;vat AI n&#225;stroje na psan&#237; k&#243;du (ChatGPT, Copilot atd.).</strong></p>
<p>Tento t&#253;den si zkus&#237;te napsat program, kter&#253; bude generovat meme obr&#225;zky ve form&#225;tu <a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/tga.html">TGA</a>. Program obdr&#382;&#237; cestu ke vstupn&#237;mu TGA obr&#225;zku, cestu ke slo&#382;ce s fontem a text, kter&#253; se m&#225; do obr&#225;zku p&#345;idat. Program pot&#233; zadan&#253; text do obr&#225;zku vykresl&#237; a obr&#225;zek ulo&#382;&#237; na disk.</p>
<p>K dispozici m&#225;te <a href="/task/UPR/2023W/BER0134/ex09_meme_generator/asset/template.tar.gz">archiv s pomocn&#253;mi soubory</a>, ve kter&#233;m naleznete n&#283;kolik TGA obr&#225;zk&#367; pro testov&#225;n&#237;, a tak&#233; slo&#382;ku s vzorov&#253;m fontem. Pokud chcete, m&#367;&#382;ete si lok&#225;ln&#283; vygenerovat TGA obr&#225;zky p&#237;smen pro vlastn&#237; font pomoc&#237; souboru <code>gen.sh</code>.</p>
<p>Program by se m&#283;l chovat takto:</p>
<ol type="1">
<li>
<p>Program na&#269;te pomoc&#237; <a href="https://mrlvsb.github.io/upr-skripta/ruzne/funkce_main.html#vstupn%C3%AD-parametry-funkce-main">parametr&#367; p&#345;&#237;kazov&#233; &#345;&#225;dky</a> t&#345;i parametry (v tomto po&#345;ad&#237;):</p>
<ul>
<li>Cestu ke vstupn&#237;mu TGA souboru (<code>input</code>)</li>
<li>Cestu k v&#253;stupn&#237;mu TGA souboru (<code>output</code>)</li>
<li>Cestu ke slo&#382;ce s fonty (<code>fonts</code>)</li>
</ul>
<p>Nap&#345;.:</p>
<pre class="console"><code>$ ./meme-generator img1.tga out.tga font</code></pre>
<ul>
<li>Pokud na vstupu programu nebudou v&#353;echny t&#345;i parametry, vypi&#353;te &#345;&#225;dek s hl&#225;&#353;kou <code>Wrong parameters</code> a ukon&#269;ete program s k&#243;dem <code>1</code>.</li>
</ul>
</li>
<li><p>Program na&#269;te z <code>input</code> vstupn&#237; TGA soubor. Pokud p&#345;i na&#269;ten&#237; dojde k chyb&#283;, vypi&#353;te &#345;&#225;dek s hl&#225;&#353;kou <code>Could not load image</code> a ukon&#269;ete program s k&#243;dem <code>1</code>.</p></li>
<li><p>Program na&#269;te ze slo&#382;ky s fonty (<code>fonts</code>) 26 TGA obr&#225;zk&#367;, jeden pro ka&#382;d&#253; znak velk&#233; anglick&#233; abecedy. Soubory jsou pojmenov&#225;ny <code>A.tga</code>, <code>B.tga</code>, <code>C.tga</code> atd. (viz slo&#382;ka <code>font</code> v <a href="/task/UPR/2023W/BER0134/ex09_meme_generator/asset/template.tar.gz">&#353;ablon&#283;</a>). Pokud p&#345;i na&#269;&#237;t&#225;n&#237; obr&#225;zk&#367; fontu dojde k chyb&#283;, program se ukon&#269;&#237; s k&#243;dem <code>1</code>.</p></li>
<li>
<p>D&#225;le program na&#269;te ze <em>standardn&#237;ho vstupu</em> &#345;&#225;dek, kter&#253; bude obsahovat dv&#283; &#269;&#237;sla odd&#283;len&#225; mezerou, <code>top</code> a <code>bottom</code>. <code>top</code> &#345;&#237;k&#225;, kolik &#345;&#225;dk&#367; textu se m&#225; vykreslit v horn&#237; &#269;&#225;sti obr&#225;zku, <code>bottom</code> ud&#225;v&#225;, kolik &#345;&#225;dk&#367; textu se m&#225; vykreslit v doln&#237; &#269;&#225;sti obr&#225;zku. D&#225;le program na&#269;te <code>top + bottom</code> &#345;&#225;dk&#367; textu ze vstupu, kter&#233; se postupn&#283; vykresl&#237; do vstupn&#237;ho TGA obr&#225;zku. P&#345;&#237;klad naleznete <a href="#p%C5%99%C3%ADklad">n&#237;&#382;e</a>.</p>
<ul>
<li>&#344;&#225;dek m&#367;&#382;e obsahovat pouze znaky mal&#233; a velk&#233; anglick&#233; abecedy a mezery, ostatn&#237; znaky ignorujte. Mal&#233; znaky anglick&#233; abecedy p&#345;ed vykreslen&#237;m p&#345;eve&#271;te na velk&#233; (dostupn&#253; font obsahuje pouze velk&#233; znaky).</li>
<li>Ka&#382;d&#253; &#345;&#225;dek bude m&#237;t maxim&#225;ln&#283; <code>100</code> znak&#367; (v&#269;etn&#283; znaku od&#345;&#225;dkov&#225;n&#237;).</li>
</ul>
</li>
<li><p>Program vykresl&#237; na&#269;ten&#233; &#345;&#225;dky do vstupn&#237;ho obr&#225;zku (<code>top</code> &#345;&#225;dk&#367; v horn&#237; &#269;&#225;sti a <code>bottom</code> &#345;&#225;dk&#367; v doln&#237; &#269;&#225;sti). Styl vykreslen&#237; je na v&#225;s, sna&#382;te se v&#353;ak co nejv&#237;ce p&#345;ibl&#237;&#382;it vzorov&#233;mu <a href="#p%C5%99%C3%ADklad">p&#345;&#237;kladu</a>.</p></li>
<li><p>Upraven&#253; vstupn&#237; TGA obr&#225;zek ulo&#382;te na disk ve form&#225;tu TGA na cestu <code>output</code> zadanou parametrem p&#345;&#237;kazov&#233; &#345;&#225;dky. P&#345;i z&#225;pisu v&#253;stupn&#237;ho TGA souboru vyu&#382;ijte hlavi&#269;ku p&#367;vodn&#237;ho vstupn&#237;ho souboru! Bude to mnohem jednodu&#353;&#353;&#237; ne&#382; se sna&#382;it z(re)konstruovat spr&#225;vnou hlavi&#269;ku ru&#269;n&#283;.</p></li>
</ol>
<h3 id="vykreslov&#225;n&#237;-p&#237;smen-fontu">Vykreslov&#225;n&#237; p&#237;smen fontu</h3>
<p>Z hodnot <code>top</code> a <code>bottom</code> vypo&#269;t&#283;te, na jak&#253;ch pozic&#237;ch se maj&#237; nach&#225;zet jednotliv&#233; &#345;&#225;dky z textu, a pot&#233; je vykreslete do na&#269;ten&#233;ho TGA obr&#225;zku.</p>
<p>Jednotliv&#233; TGA obr&#225;zky znak&#367; fontu obsahuj&#237; &#269;ern&#233; pozad&#237; (<code>RGB (0, 0, 0)</code>) a b&#237;l&#233; pop&#345;ed&#237;. P&#345;i kop&#237;rov&#225;n&#237; znaku do pixel&#367; vstupn&#237;ho obr&#225;zku ignorujte &#269;ern&#233; pixely pozad&#237; a kop&#237;rujte tedy jenom pixely, kter&#233; nejsou &#269;ern&#233;. Jinak byste ve v&#253;sledn&#233;m obr&#225;zku m&#283;li kolem ka&#382;d&#233;ho znaku ru&#353;iv&#253; &#269;ern&#253; obd&#233;ln&#237;k.</p>
<p>P&#345;edpokl&#225;dejte, &#382;e v&#253;&#353;ka ka&#382;d&#233;ho znaku ve fontu je <code>34</code> pixel&#367;, &#353;&#237;&#345;ka je pro ka&#382;d&#253; znak r&#367;zn&#225;. P&#345;i vykreslov&#225;n&#237; tak mus&#237;te br&#225;t v potaz &#353;&#237;&#345;ku vykreslovan&#253;ch znak&#367;, a korektn&#283; se p&#345;i vypisov&#225;n&#237; znak&#367; v obr&#225;zku horizont&#225;ln&#283; posouvat, aby se jednotliv&#233; znaky nep&#345;ekr&#253;valy.</p>
<h3 id="pozn&#225;mky">Pozn&#225;mky</h3>
<ul>
<li>
<strong>Pou&#382;it&#237; <a href="https://mrlvsb.github.io/upr-skripta/c/pole/staticka_pole.html#konstantn%C3%AD-velikost-statick%C3%A9ho-pole">VLA</a> je zak&#225;z&#225;no</strong>.</li>
<li>V implementaci &#250;lohy si vhodn&#283; nadefinujte si vlastn&#237; datov&#233; typy, nap&#345;. pro reprezentaci TGA obr&#225;zku v pam&#283;ti, a&#357; se v&#225;m s obr&#225;zky dob&#345;e pracuje. Vytvo&#345;te tak&#233; sadu funkc&#237; pro manipulaci s TGA obr&#225;zkem, a&#357; nem&#225;te ve&#353;ker&#253; k&#243;d ve funkci <code>main</code>.</li>
<li>Nemus&#237;te podporovat &#269;ernob&#237;l&#233; TGA obr&#225;zky ani TGA obr&#225;zky s pr&#367;hlednost&#237; (RGBA). Pro tuto &#250;lohu p&#345;edpokl&#225;dejte, &#382;e v&#353;echny TGA obr&#225;zky maj&#237; barevnou hloubku 24 bit&#367; (RGB) a &#382;e jejich sou&#345;adn&#253; syst&#233;m za&#269;&#237;n&#225; v lev&#233;m horn&#237;m rohu (tj. ne&#345;e&#353;te 5. bit v popisova&#269;i ani hodnoty po&#269;&#225;tku v hlavi&#269;ce).</li>
<li>P&#345;i pr&#225;ci s &#345;et&#283;zci &#269;i pixely budete nar&#225;&#382;et na pam&#283;&#357;ov&#233; chyby. Pou&#382;&#237;vejte <a href="#kontrola-pam%C4%9B%C5%A5ov%C3%BDch-chyb">Address sanitizer nebo Valgrind</a>!</li>
<li>P&#345;i &#345;e&#353;en&#237; t&#233;to &#250;lohy bude velmi u&#382;ite&#269;n&#233; vyu&#382;&#237;t <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#krokov%C3%A1n%C3%AD">debugger</a> VSCode.</li>
<li>I p&#345;i lad&#283;n&#237;/krokov&#225;n&#237; si m&#367;&#382;ete na vstup programu <a href="https://code.visualstudio.com/docs/editor/debugging#_redirect-inputoutput-tofrom-the-debug-target">p&#345;esm&#283;rovat</a> soubor, abyste nemuseli vstup neust&#225;le ps&#225;t ru&#269;n&#283;.</li>
</ul>
<p>D&#233;lka referen&#269;n&#237;ho &#345;e&#353;en&#237;: ~190 &#345;&#225;dk&#367;</p>
<h3 id="p&#345;&#237;klad">P&#345;&#237;klad</h3>
<ul>
<li>
<p>Spu&#353;t&#283;n&#237; programu (obr&#225;zek <code>img1.tga</code> m&#367;&#382;ete naleznout v &#353;ablon&#283;):</p>
<pre class="console"><code>$ ./meme-generator img1.tga out.tga font
2 2
I dont always do
memes
but when i do
i do them in C</code></pre>
<p>Uk&#225;zkov&#253; v&#253;stup (soubor <code>out.tga</code>):</p>
<p><img src="img/meme1.png"></p>
</li>
<li>
<p>Spu&#353;t&#283;n&#237; programu (obr&#225;zek <code>img2.tga</code> m&#367;&#382;ete naleznout v &#353;ablon&#283;):</p>
<pre class="console"><code>$ ./meme-generator img2.tga out.tga font
0 1
you to read skripta</code></pre>
<p>Uk&#225;zkov&#253; v&#253;stup (soubor <code>out.tga</code>):</p>
<p><img src="img/meme2.png"></p>
</li>
</ul>
<h3 id="u&#382;ite&#269;n&#233;-funkce">U&#382;ite&#269;n&#233; funkce</h3>
<ul>
<li>
<a href="https://devdocs.io/c/io/fgets"><code>fgets</code></a> - na&#269;ten&#237; &#345;&#225;dku ze vstupu do &#345;et&#283;zce (pole znak&#367;).</li>
<li>
<a href="https://devdocs.io/c/string/byte/atoi"><code>atoi</code></a> - p&#345;eveden&#237; &#345;et&#283;zce obsahuj&#237;c&#237;ho &#269;&#237;slice na cel&#233; &#269;&#237;slo (<code>int</code>).</li>
<li>
<a href="https://devdocs.io/c/io/fprintf"><code>snprintf</code></a> - vyps&#225;n&#237; form&#225;tovan&#253;ch dat do &#345;et&#283;zce v pam&#283;ti.</li>
</ul>
<h3 id="bonusov&#253;-&#250;kol">Bonusov&#253; &#250;kol</h3>
<p>Vykreslete text do obr&#225;zku tak, aby byl vycentrovan&#253; (viz p&#345;&#237;klady naho&#345;e). Pohrajte si s grafick&#253;m stylem vykreslen&#237;, zkuste nap&#345;. d&#225;t ka&#382;d&#233;mu vykreslen&#233;mu znaku jinou barvu.</p>
<h3 id="testov&#225;n&#237;-programu">Testov&#225;n&#237; programu</h3>
<p>Vlastn&#237; TGA obr&#225;zky pro testov&#225;n&#237; si m&#367;&#382;ete vygenerovat nap&#345;. pomoc&#237; programu <a href="https://linuxopsys.com/topics/install-latest-imagemagick-on-ubuntu-20-04">ImageMagick</a>:</p>
<pre class="console"><code>$ convert myimage.jpg -alpha off -define tga:image-origin=TopLeft -depth 8 -type TrueColor myimage.tga</code></pre>
<h3 id="p&#345;esm&#283;rov&#225;n&#237;-vstupu">P&#345;esm&#283;rov&#225;n&#237; vstupu</h3>
<p>Abyste nemuseli neust&#225;le ru&#269;n&#283; zad&#225;vat text z kl&#225;vesnice p&#345;i testov&#225;n&#237; programu, m&#367;&#382;ete data na vstup programu <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstupavystup.html#standardn%C3%AD-souborov%C3%A9-deskriptory">p&#345;esm&#283;rovat</a> ze souboru:</p>

# spu&#353;t&#283;n&#237; souboru, p&#345;esm&#283;rov&#225;n&#237; souboru test-1/stdin na vstup programu

<pre class="sh"><code># p&#345;eklad programu
gcc -g -fsanitize=address main.c -o program


./program img1.tga out.tga font &lt; test-1/stdin
</code></pre>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
</div>