<div>
<p>Tento t&#253;den si zkus&#237;te vytvo&#345;it interaktivn&#237; grafickou aplikaci pomoc&#237; knihovny <a href="https://www.libsdl.org/">SDL</a>, a tak&#233; si vyzkou&#353;&#237;te pr&#225;ci s dynamicky rostouc&#237;m polem.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/sdl.html">SDL</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/ruzne/dynamicky_rostouci_pole.html">Dynamicky rostouc&#237; pole</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/ruzne/nahodna_cisla.html">Generov&#225;n&#237; n&#225;hodn&#253;ch &#269;&#237;sel</a></li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/sdl.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny. Soubory <code>dynamic_array.h/c</code> a <code>sdl.h/c</code> nemus&#237;te nahr&#225;vat, Kelvin je automaticky p&#345;ikop&#237;ruje do slo&#382;ky s va&#353;&#237;m zdrojov&#253;m k&#243;dem.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu. Je zak&#225;z&#225;no sd&#237;let sv&#233; &#345;e&#353;en&#237; s ostatn&#237;mi, opisovat od ostatn&#237;ch, nechat si od ostatn&#237;ch diktovat k&#243;d a pou&#382;&#237;vat AI n&#225;stroje na psan&#237; k&#243;du (ChatGPT, Copilot atd.).</strong></p>
<details>
<p><summary>Pro&#269; zakazujeme pou&#382;it&#237; AI pro psan&#237; k&#243;du?</summary></p>
<p>&#218;lohy zad&#225;van&#233; v UPR jsou svou strukturou velmi jednoduch&#233;, a daj&#237; se obvykle vy&#345;e&#353;it pomoc&#237; n&#283;kolika des&#237;tek &#345;&#225;dk&#367; k&#243;du. Dne&#353;n&#237; jazykov&#233; modely jsou schopny tyto &#250;lohy vy&#345;e&#353;it v podstat&#283; na prvn&#237; dobrou, t&#233;m&#283;&#345; nebo zcela dokonale. Pokud tedy student pouze nakop&#237;ruje zad&#225;n&#237; do AI modelu, a pot&#233; zkop&#237;ruje v&#253;sledek do Kelvina, tak se samoz&#345;ejm&#283; nic nenau&#269;&#237;. C&#237;lem UPR je nau&#269;it se pochopit z&#225;klady toho, jak funguje po&#269;&#237;ta&#269; a pam&#283;&#357;, jak funguje programovac&#237; jazyk C a jak pomoc&#237; n&#283;j vytv&#225;&#345;et jednoduch&#233; programy, a toho lze dos&#225;hnout pouze t&#237;m, &#382;e budete hodn&#283; programovat a &#345;e&#353;it hodn&#283; &#250;loh.</p>
<p>Probl&#233;m je v tom, &#382;e AI v&#225;m sice vy&#345;e&#353;&#237; jednoduch&#233; probl&#233;my, ale u slo&#382;it&#283;j&#353;&#237;ch &#250;loh (kter&#233; se budou vyskytovat nap&#345;. v navazuj&#237;c&#237;ch p&#345;edm&#283;tech, nebo pozd&#283;ji v pr&#225;ci) u&#382; si bu&#271; nebude v&#283;d&#283;t rady, anebo budete muset jej&#237; v&#253;stup alespo&#328; &#269;&#225;ste&#269;n&#283; upravit. A abyste mohli n&#283;co upravit, tak tomu mus&#237;te rozum&#283;t, a mus&#237;te b&#253;t schopni to nez&#225;visle vytvo&#345;it samostatn&#283;. A k tomu je pot&#345;eba j&#237;t postupn&#283;, nejprve se nau&#269;it &#345;e&#353;it jednodu&#353;&#353;&#237; &#250;lohy, a pak postupn&#283; p&#345;ech&#225;zet na slo&#382;it&#283;j&#353;&#237;. Pokud jednoduch&#233; &#250;lohy vy&#345;e&#353;&#237;te pomoc&#237; AI, a u slo&#382;it&#283;j&#353;&#237;ch AI sel&#382;e, tak nebudete v&#367;bec v&#283;d&#283;t, jak tyto slo&#382;it&#283;j&#353;&#237; &#250;lohy &#345;e&#353;it.</p>
Z toho d&#367;vodu nedovolujeme pou&#382;&#237;vat AI v re&#382;imu &#8220;copy &amp; paste&#8221;. Pokud pou&#382;ijete AI pro &#8220;konverzaci&#8221;, nap&#345;. pro zji&#353;t&#283;n&#237;, jestli by V&#225;&#353; k&#243;d ne&#353;el zlep&#353;it, nebo jestli v n&#283;m nen&#237; chyba, tak je to v po&#345;&#225;dku. Pokud ale pouze do ChatGPT nakop&#237;rujete zad&#225;n&#237;, a pot&#233; zkop&#237;rujete vygenerovan&#253; k&#243;d, tak se opravdu nic nenau&#269;&#237;te.
</details>
<p><br></p>
<p>Napi&#353;te program pomoc&#237; knihovny <code>SDL</code>, kter&#253; umo&#382;n&#237; u&#382;ivatelovi vytv&#225;&#345;et vlo&#269;ky, kter&#233; budou &#8220;sn&#283;&#382;it&#8221; (padat dol&#367;). Vlo&#269;ky se vytvo&#345;&#237; s <a href="https://mrlvsb.github.io/upr-skripta/ruzne/nahodna_cisla.html">n&#225;hodnou</a> velikost&#237;, rychlost&#237; a rotac&#237;. Vlo&#269;ky budou padat dol&#367; a p&#345;itom se ot&#225;&#269;et. Rychlost pad&#225;n&#237; i ot&#225;&#269;en&#237; bude ovliv&#328;ovat n&#225;hodn&#283; vygenerovan&#225; rychlost vlo&#269;ky. Jakmile se vlo&#269;ka dostane za doln&#237; okraj obrazovky, tak se uvoln&#237; z pam&#283;ti a p&#345;estane se vykreslovat.</p>
<p>M&#367;&#382;ete si vybrat, jestli se vlo&#269;ky budou vytv&#225;&#345;et po kliknut&#237; my&#353;i nebo p&#345;i pohybu my&#353;i (viz <a href="#uk%C3%A1zka-fungov%C3%A1n%C3%AD-programu">uk&#225;zka</a> n&#237;&#382;e).</p>
<p>N&#225;vod k implementaci:</p>
<ul>
<li>V&#353;e, co v programu budete vykreslovat, si mus&#237;te uchov&#225;vat a n&#283;jak reprezentovat v pam&#283;ti. Vytvo&#345;te si tedy reprezentaci vlo&#269;ky, kter&#225; si bude pamatovat v&#353;echny pot&#345;ebn&#233; &#250;daje (viz zad&#225;n&#237; v&#253;&#353;e).</li>
<li>Vlo&#269;ky si uchov&#225;vejte v dynamicky rostouc&#237;m poli.</li>
<li>P&#345;i v&#253;skytu n&#283;jak&#233; ud&#225;losti my&#353;i (klik &#269;i pohnut&#237;) vytvo&#345;te novou vlo&#269;ku, a p&#345;idejte ji do pole.</li>
<li>Pokud naopak n&#283;kter&#225; vlo&#269;ka zmiz&#237; z obrazovky, tak ji z pole sma&#382;te.</li>
<li>V ka&#382;d&#233;m sn&#237;mku aplikace vykreslujte v&#353;echny vlo&#269;ky na jejich sou&#269;asn&#233; pozici.</li>
</ul>
<p>D&#233;lka referen&#269;n&#237;ho &#345;e&#353;en&#237;: ~200 &#345;&#225;dk&#367; (v&#269;etn&#283; bonusu).</p>
<h2 id="uk&#225;zka-fungov&#225;n&#237;-programu">Uk&#225;zka fungov&#225;n&#237; programu:</h2>
<ul>
<li>
<p>Vytv&#225;&#345;en&#237; vlo&#269;ky po kliknut&#237; my&#353;&#237;:</p>
<p><video width="600" height="480" controls> <source src="img/snowflakes.mp4" type="video/mp4"> Your browser does not support the video tag. </source></video></p>
</li>
<li>
<p>Vytv&#225;&#345;en&#237; vlo&#269;ky p&#345;i pohybu my&#353;i:</p>
<video width="600" height="480" controls>
<source src="img/snowflakes_motion.mp4" type="video/mp4">
<p>Your browser does not support the video tag. </p></source></video>
</li>
</ul>
<h2 id="pozn&#225;mky-k-implementaci">Pozn&#225;mky k implementaci</h2>
<p>Pro vlo&#269;ku pou&#382;ijte <a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/assets/snowflake.svg">tento obr&#225;zek</a>.</p>
<p>Pro ukl&#225;d&#225;n&#237; vlo&#269;ek se v&#225;m bude hodit <a href="https://mrlvsb.github.io/upr-skripta/ruzne/dynamicky_rostouci_pole.html">dynamicky rostouc&#237; pole</a>. M&#367;&#382;ete pou&#382;&#237;t n&#225;sleduj&#237;c&#237; implementaci:</p>
<ul>
<li><a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/template/dynamic_array.h"><code>dynamic_array.h</code></a></li>
<li><a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/template/dynamic_array.c"><code>dynamic_array.c</code></a></li>
</ul>
<p>M&#367;&#382;ete tak&#233; vyu&#382;&#237;t tyto pomocn&#233; funkce pro inicializaci <code>SDL</code>:</p>
<ul>
<li><a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/template/sdl.h"><code>sdl.h</code></a></li>
<li><a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/template/sdl.c"><code>sdl.c</code></a></li>
</ul>
<p>V&#353;echny tyto zdrojov&#233; soubory si m&#367;&#382;ete st&#225;hnout v <a href="/task/UPR/2023W/BER0134/ex10_sdl/asset/template.tar.gz">tomto archivu</a>.</p>
<p>Pokud pou&#382;&#237;v&#225;te WSL, tak ve skriptech naleznete <a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/sdl.html#zprovozn%C4%9Bn%C3%AD-sdl-pod-wsl">n&#225;vod</a>, jak zprovoznit SDL pod WSL. Pokud v&#225;m bude aplikace vytv&#225;&#345;et n&#283;jak&#233; grafick&#233; artefakty nebo nebude vykreslov&#225;n&#237; fungovat (zkuste si hello world SDL aplikaci ze skript), tak zkuste pou&#382;&#237;t Emulaci X serveru.</p>
<h2 id="bonusov&#253;-&#250;kol">Bonusov&#253; &#250;kol</h2>
<p>P&#345;idejte ke vlo&#269;ce &#8220;ocas&#8221; pomoc&#237; tzv. <strong>&#269;&#225;sticov&#253;ch efekt&#367;</strong> (<em>particle effects</em>). Jednou za &#269;as m&#367;&#382;ete na pozici vlo&#269;ky vytvo&#345;it n&#283;kolik bod&#367;, kter&#233; budou postupn&#283; &#8220;zhas&#237;nat&#8221; (a&#382; zhasnou &#250;pln&#283;, tak zmiz&#237;). Podle toho, jak dlouho u&#382; existuj&#237;, m&#283;&#328;te jejich <a href="https://stackoverflow.com/questions/31770785/changing-alpha-value-in-sdl-setrenderdrawcolor-doesnt-effect-anything-sdl2">alfa kan&#225;l</a>, abyste dos&#225;hli postupn&#233;ho &#8220;mizen&#237;&#8221;:</p>
<video width="600" height="480" controls>
<source src="img/snowflakes_trail.mp4" type="video/mp4">
<p>Your browser does not support the video tag. </p></source></video>
<h2 id="u&#382;ite&#269;n&#233;-funkce">U&#382;ite&#269;n&#233; funkce</h2>
<ul>
<li>
<a href="https://hg.libsdl.org/SDL_image/file/ca95d0e31aec/IMG.c#l209"><code>IMG_LoadTexture</code></a>: na&#269;ten&#237; obr&#225;zku (textury) z disku.</li>
<li>
<a href="https://wiki.libsdl.org/SDL_RenderCopyEx"><code>SDL_RenderCopyEx</code></a>: vykreslen&#237; obr&#225;zku (textury) s danou velikost&#237; a rotac&#237; do kresl&#237;tka.</li>
</ul>
<h2 id="p&#345;eklad-programu">P&#345;eklad programu</h2>
<p>Pro pou&#382;it&#237; SDL si tuto knihovnu budete muset nainstalovat, a pot&#233; p&#345;ilinkovat k va&#353;emu programu. N&#225;vod naleznete ve <a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/sdl.html#instalace-sdl">skriptech</a>. Budete si tak&#233; muset ke sv&#233;mu programu p&#345;ikompilovat soubor <code>dynamic_array.c</code>.</p>
<p>Takto m&#367;&#382;ete p&#345;elo&#382;it v&#353;e najednou. Nen&#237; to ide&#225;ln&#237; zp&#367;sob (viz <a href="https://mrlvsb.github.io/upr-skripta/c/modularizace/linker.html#pro%C4%8D-takto-slo%C5%BEit%C4%9B">skripta</a>), ale pro tento jednoduch&#253; program to je nejjednodu&#353;&#353;&#237; volba:</p>
<pre class="bash"><code>$ gcc main.c sdl.c dynamic_array.c -lSDL2 -lSDL2_image -g -fsanitize=address -omain</code></pre>
<p>M&#367;&#382;ete tak&#233; vyu&#382;&#237;t <a href="https://mrlvsb.github.io/upr-skripta/c/aplikovane_ulohy/sdl.html#p%C5%99ilinkov%C3%A1n%C3%AD-knihovny-sdl"><code>CMake</code></a>, aby va&#353;e IDE v&#283;d&#283;lo o tom, &#382;e knihovnu SDL pou&#382;&#237;v&#225;te, a nab&#237;zelo v&#225;m tak dopl&#328;ov&#225;n&#237; k&#243;du.</p>
<h3 id="kontrola-pam&#283;&#357;ov&#253;ch-chyb">Kontrola pam&#283;&#357;ov&#253;ch chyb</h3>
<p>P&#345;i p&#345;ekladu pou&#382;&#237;vejte <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#address-sanitizer">Address sanitizer</a> nebo <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#valgrind">Valgrind</a>, abyste mohli rychle odhalit (t&#233;m&#283;&#345; nevyhnuteln&#233;) <a href="https://mrlvsb.github.io/upr-skripta/caste_chyby/pametove_chyby.html">pam&#283;&#357;ov&#233; chyby</a>.</p>
<p>P&#345;i pou&#382;it&#237; knihovny <code>SDL</code> se m&#367;&#382;e st&#225;t, &#382;e Address sanitizer ohl&#225;s&#237; n&#283;jak&#233; memory leaky, kter&#233; nebudou souviset s va&#353;&#237;m programem. Ty m&#367;&#382;ete ignorovat.</p>
</div>