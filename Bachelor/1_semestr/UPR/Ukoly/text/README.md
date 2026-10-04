<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak m&#367;&#382;eme v jazyce <em>C</em> pracovat s textem (znaky, sekvencemi znak&#367;), a tak&#233; jak textov&#225; data na&#269;&#237;tat a vypisovat.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/text/znaky.html">Znaky</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/text/retezce.html">&#344;et&#283;zce</a></li>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/c/text/vstupavystup.html">Vstup a v&#253;stup</a>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html">Vstup</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/text/vystup.html">V&#253;stup</a></li>
</ul>
</li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/text.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Odevzd&#225;vejte jeden soubor s p&#345;&#237;ponou <code>.c</code>. &#218;lohy odevzdan&#233; v archivu <code>.rar</code> nebo s jinou p&#345;&#237;ponou nebudou akceptov&#225;ny.</strong></p>
<p><strong>Na &#250;loze pracujte samostatn&#283;. Pokud zjist&#237;me, &#382;e jste nepracovali na &#250;loze samostatn&#283;, budou v&#225;m ud&#283;leny z&#225;porn&#233; body, p&#345;&#237;padn&#283; budete vylou&#269;eni z p&#345;edm&#283;tu. Je zak&#225;z&#225;no sd&#237;let sv&#233; &#345;e&#353;en&#237; s ostatn&#237;mi, opisovat od ostatn&#237;ch, nechat si od ostatn&#237;ch diktovat k&#243;d a pou&#382;&#237;vat AI n&#225;stroje na psan&#237; k&#243;du (ChatGPT, Copilot atd.).</strong></p>
<p>Tento t&#253;den si zkus&#237;te napsat program, kter&#253; bude prov&#225;d&#283;t manipulaci s textem. Program na&#269;te sadu &#345;&#225;dk&#367; se slovy, ka&#382;d&#233; slovo ze vstupu &#8220;znormalizuje&#8221;, a pot&#233; vyp&#237;&#353;e &#345;&#225;dek se znormalizovan&#253;mi slovy, a tak&#233; jednoduchou statistiku.</p>
<p>Program by se m&#283;l chovat takto:</p>
<ol type="1">
<li>Na&#269;te ze vstupu jedno &#269;&#237;slo <code>n</code>, kter&#233; ud&#225;v&#225;, kolik &#345;&#225;dk&#367; se slovy bude ve vstupu n&#225;sledovat.
<ul>
<li>&#344;&#225;dek s &#269;&#237;slem <code>n</code> bude m&#237;t maxim&#225;ln&#283; <code>50</code> znak&#367; (v&#269;etn&#283; znaku od&#345;&#225;dkov&#225;n&#237;).</li>
</ul>
</li>
<li>D&#225;le na&#269;te ze vstupu postupn&#283; <code>n</code> &#345;&#225;dk&#367;.
<ul>
<li>Ka&#382;d&#253; &#345;&#225;dek bude m&#237;t maxim&#225;ln&#283; <code>50</code> znak&#367; (v&#269;etn&#283; znaku od&#345;&#225;dkov&#225;n&#237;).
<ul>
<li>Jak velkou pam&#283;&#357; mus&#237;te m&#237;t v programu p&#345;ipravenou, abyste mohli na&#269;&#237;st 50 znak&#367;?</li>
</ul>
</li>
<li>&#344;&#225;dky budou obsahovat slova odd&#283;len&#225; mezerami. Libovoln&#253; po&#269;et mezer m&#367;&#382;e b&#253;t na za&#269;&#225;tku &#345;&#225;dku, na konci &#345;&#225;dku, nebo mezi slovy, dv&#283; slova jsou v&#353;ak v&#382;dy rozd&#283;lena alespo&#328; jednou mezerou.</li>
<li>Ka&#382;d&#233; slovo se skl&#225;d&#225; pouze z p&#237;smen mal&#233; a velk&#233; anglick&#233; abecedy (<code>a-z</code>, <code>A-Z</code>). Slovo neobsahuje mezeru.</li>
</ul>
</li>
<li>Pro ka&#382;d&#253; &#345;&#225;dek vypi&#353;te jeho znormalizovanou verzi (viz <a href="#normalizovan%C3%BD-%C5%99%C3%A1dek">n&#237;&#382;e</a>), a tak&#233; jednoduchou statistiku:
<ul>
<li>Kolik bylo v &#345;&#225;dku p&#367;vodn&#283; znak&#367; mal&#233; anglick&#233; abecedy, a kolik jich je v znormalizovan&#233;m &#345;&#225;dku.</li>
<li>Kolik bylo v &#345;&#225;dku p&#367;vodn&#283; znak&#367; velk&#233; anglick&#233; abecedy, a kolik jich je v znormalizovan&#233;m &#345;&#225;dku.</li>
<li>Kolik bylo v &#345;&#225;dku p&#367;vodn&#283; mezer, a kolik jich je v znormalizovan&#233;m &#345;&#225;dku. Mezi ka&#382;d&#253;mi dv&#283;ma vypsan&#253;mi &#345;&#225;dky se statistikou vypi&#353;te jeden pr&#225;zdn&#253; &#345;&#225;dek. Viz <a href="#tests">Testy</a>.</li>
</ul>
</li>
</ol>
<h3 id="normalizovan&#253;-&#345;&#225;dek">Normalizovan&#253; &#345;&#225;dek</h3>
<p>Normalizovan&#253; &#345;&#225;dek obsahuje v&#353;echna slova p&#367;vodn&#237;ho &#345;&#225;dku ve stejn&#233;m po&#345;ad&#237;, ale z&#225;rove&#328; plat&#237;, &#382;e:</p>
<ul>
<li>Neobsahuje &#382;&#225;dn&#233; mezery na za&#269;&#225;tku ani na konci.</li>
<li>Mezi ka&#382;d&#253;mi dv&#283;ma slovy je pr&#225;v&#283; jedna mezera.</li>
<li>Ka&#382;d&#233; slovo je normalizovan&#233;.</li>
</ul>
<h3 id="normalizovan&#233;-slovo">Normalizovan&#233; slovo</h3>
<p>Normalizovan&#233; slovo m&#225; takov&#233;to pravidla:</p>
<ul>
<li>Pokud v p&#367;vodn&#237;m slov&#283; bylo alespo&#328; jedno p&#237;smeno velk&#233; anglick&#233; abecedy, tak v normalizovan&#233;m slov&#283; bude prvn&#237; znak velk&#253;, a v&#353;echny ostatn&#237; znaky mal&#233; (<code>xBc</code> -&gt; <code>Xbc</code>, <code>ABC</code> -&gt; <code>Abc</code>).</li>
<li>Pokud v p&#367;vodn&#237;m slov&#283; nebylo ani jedno p&#237;smeno velk&#233; anglick&#233; abecedy, tak v normalizovan&#233;m slov&#283; budou v&#353;echny znaky velk&#233; (<code>xbc</code> -&gt; <code>XBC</code>, <code>abc</code> -&gt; <code>ABC</code>).</li>
</ul>
<p>D&#225;le, pokud se po proveden&#237; normalizace popsan&#233; v&#253;&#353;e vyskytne ve slovu v&#237;ce stejn&#253;ch znak&#367; za sebou, tak se z t&#233;to sekvence znak&#367; zachov&#225; pouze prvn&#237; znak:</p>
<ul>
<li>
<code>xxx</code> -&gt; <code>X</code>
</li>
<li>
<code>xXx</code> -&gt; <code>Xx</code>
</li>
<li>
<code>aabcc</code> -&gt; <code>ABC</code>
</li>
</ul>
<p>P&#345;&#237;klad vstupu:</p>
<pre><code>3
  uwu wRiTE
HoW     uNIveRsITY brittle
Heleh xxxXX</code></pre>
<p>Odpov&#237;daj&#237;c&#237; v&#253;stup:</p>
<pre class="text"><code>UWU Write
lowercase: 5 -&gt; 4
uppercase: 3 -&gt; 4
spaces: 3 -&gt; 1

How University BRITLE
lowercase: 12 -&gt; 11
uppercase: 8 -&gt; 8
spaces: 6 -&gt; 2

Heleh Xx
lowercase: 7 -&gt; 5
uppercase: 3 -&gt; 2
spaces: 1 -&gt; 1</code></pre>
<h3 id="pozn&#225;mky-k-&#345;e&#353;en&#237;">Pozn&#225;mky k &#345;e&#353;en&#237;</h3>
<ul>
<li>Nemus&#237;te si v&#353;echny &#345;&#225;dky ulo&#382;it do jednoho velk&#233;ho pole. Sta&#269;&#237; &#345;&#225;dky zpracov&#225;vat postupn&#283;, jeden po druh&#233;m.</li>
<li>Pro &#345;e&#353;en&#237; &#250;lohy nen&#237; pot&#345;eba dynamick&#225; alokace pam&#283;ti, ani v&#237;ce pol&#237;. Sta&#269;&#237; vytvo&#345;it jedno pole na z&#225;sobn&#237;ku, pro &#345;&#225;dek.</li>
<li>
<strong>Pro ve&#353;ker&#233; na&#269;&#237;t&#225;n&#237; vstupn&#237;ch dat (i &#269;&#237;sla <code>n</code>) pou&#382;&#237;vejte funkci <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html#na%C4%8Dten%C3%AD-%C5%99%C3%A1dku"><code>fgets</code></a>. Pokud budete kombinovat <code>scanf</code> a <code>fgets</code>, <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html#zpracov%C3%A1n%C3%AD-b%C3%ADl%C3%BDch-znak%C5%AF">budete nar&#225;&#382;et na probl&#233;my</a>. Proto rad&#283;ji v&#353;e na&#269;&#237;tejte funkc&#237; <code>fgets</code> a pro p&#345;evod na &#269;&#237;slo vyu&#382;ijte nap&#345;. funkci <a href="https://devdocs.io/c/string/byte/atoi"><code>atoi</code></a></strong>.</li>
<li>D&#225;vejte si u funkce <code>fgets</code> pozor na to, &#382;e znak od&#345;&#225;dkov&#225;n&#237; je tak&#233; sou&#269;&#225;st&#237; vstupu! Viz <a href="https://mrlvsb.github.io/upr-skripta/c/text/vstup.html#na%C4%8Dten%C3%AD-%C5%99%C3%A1dku">skripta</a>.</li>
<li>P&#345;i pr&#225;ci s &#345;et&#283;zci budete nar&#225;&#382;et na pam&#283;&#357;ov&#233; chyby. Pou&#382;&#237;vejte n&#225;stroje pro <a href="#kontrola-pam%C4%9B%C5%A5ov%C3%BDch-chyb">detekci pam&#283;&#357;ov&#253;ch chyb</a>! P&#345;i &#345;e&#353;en&#237; t&#233;to &#250;lohy bude tak&#233; velmi u&#382;ite&#269;n&#233; vyu&#382;&#237;t <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html#krokov%C3%A1n%C3%AD">debugger</a> VSCode. I p&#345;i lad&#283;n&#237;/krokov&#225;n&#237; si m&#367;&#382;ete na vstup programu <a href="https://code.visualstudio.com/docs/editor/debugging#_redirect-inputoutput-tofrom-the-debug-target">p&#345;esm&#283;rovat</a> soubor, abyste nemuseli vstup neust&#225;le ps&#225;t ru&#269;n&#283;.</li>
</ul>
<p>D&#233;lka referen&#269;n&#237;ho &#345;e&#353;en&#237;: ~120 &#345;&#225;dk&#367;</p>
<h3 id="u&#382;ite&#269;n&#233;-funkce">U&#382;ite&#269;n&#233; funkce</h3>
<ul>
<li>
<a href="https://devdocs.io/c/io/fgets"><code>fgets</code></a> - na&#269;ten&#237; &#345;&#225;dku ze vstupu do &#345;et&#283;zce (pole znak&#367;).</li>
<li>
<a href="https://devdocs.io/c/string/byte/atoi"><code>atoi</code></a> - p&#345;eveden&#237; &#345;et&#283;zce obsahuj&#237;c&#237;ho &#269;&#237;slice na &#269;&#237;slo (<code>int</code>).</li>
<li>
<a href="https://devdocs.io/c/string/byte/strlen"><code>strlen</code></a> - zji&#353;t&#283;n&#237; d&#233;lky &#345;et&#283;zce.</li>
</ul>
<h3 id="bonusov&#233;-&#250;koly">Bonusov&#233; &#250;koly</h3>
<ul>
<li>Pro p&#345;evod &#345;et&#283;zce na &#269;&#237;slo nepou&#382;&#237;vejte <code>atoi</code> ani <code>strtol</code>, ale naprogramujte si pro to vlastn&#237; funkci.</li>
<li>Naprogramujte si sami v&#353;echny nutn&#233; funkce pro pr&#225;ci s &#345;et&#283;zci. Krom&#283; funkce <code>fgets</code> tedy nepou&#382;&#237;vejte nic ze standardn&#237; knihovny - <code>strlen</code>, <code>tolower</code>, <code>islower</code>, <code>isdigit</code>, <code>atoi</code> atd.</li>
</ul>
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
