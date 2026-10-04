<div>
<p>Tento t&#253;den si uk&#225;&#382;eme, jak vytv&#225;&#345;et zaj&#237;mav&#283;j&#353;&#237; programy pomoc&#237; cykl&#367; a podm&#237;nek.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/rizeni_toku/podminky.html">Podm&#237;nky</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/rizeni_toku/cykly.html">Cykly</a></li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/podminky_a_cykly.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p>Va&#353;&#237;m &#250;kolem bude vytvo&#345;it program, kter&#253; bude um&#283;t vypisovat r&#367;zn&#233; geometrick&#233; tvary na v&#253;stup. P&#345;i spu&#353;t&#283;n&#237; program na&#269;te ze standardn&#237;ho vstupu (z termin&#225;lu) t&#345;i &#269;&#237;sla (<code>obrazec</code>, <code>A</code> a <code>B</code>). V z&#225;vislosti na hodnot&#283; &#269;&#237;sla <code>obrazec</code> pot&#233; program vykresl&#237; dan&#253; geometrick&#253; tvar, jeho&#382; tvar bude ovlivn&#283;n hodnotami parametr&#367; <code>A</code> a <code>B</code>.</p>
<p><strong>Pokud byste m&#283;li k &#250;loze jak&#233;koliv dotazy, napi&#353;te pros&#237;m na Discord do m&#237;stnosti <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>, r&#225;di v&#353;e vysv&#283;tl&#237;me.</strong></p>
<h3 id="na&#269;&#237;t&#225;n&#237;-vstupu">Na&#269;&#237;t&#225;n&#237; vstupu</h3>
<p>Pro na&#269;ten&#237; t&#345;&#237; &#269;&#237;sel m&#367;&#382;ete vyu&#382;&#237;t tuto &#353;ablonu:</p>
<pre class="c"><code>#include &lt;stdio.h&gt;

int main() {
    int obrazec = 0;
    int a = 0;
    int b = 0;
    
    scanf("%d%d%d", &amp;obrazec, &amp;a, &amp;b);
    
    // Zde m&#367;&#382;ete pracovat s prom&#283;nn&#253;mi `obrazec`, `a`, `b`, kter&#233; byly na&#269;teny ze vstupu programu

    return 0;
}</code></pre>
<p>Po spu&#353;t&#283;n&#237; programu m&#367;&#382;ete vstup (t&#345;i &#269;&#237;sla odd&#283;len&#225; mezerou) v termin&#225;lu:</p>
<pre class="bash"><code>$ ./program
0 3 3</code></pre>
<p>P&#345;&#237;padn&#283; m&#367;&#382;ete vstup programu ulo&#382;it do souboru, a pot&#233; si tento soubor takzvan&#283; p&#345;esm&#283;rovat na vstup programu. Pokud byste si nap&#345;. text <code>0 3 3</code> ulo&#382;ili do souboru s n&#225;zvem <code>test-square.in</code>, tak tento soubor m&#367;&#382;ete p&#345;esm&#283;rovat na vstup programu takto:</p>
<pre class="bash"><code>$ ./program &lt; test-square.in</code></pre>
<h3 id="typy-obrazc&#367;">Typy obrazc&#367;</h3>
<p>N&#237;&#382;e naleznete seznam obrazc&#367;, kter&#233; by m&#283;l V&#225;&#353; program podporovat.</p>
<ul>
<li>
<p>(0.5b) Pokud m&#225; <code>obrazec</code> hodnotu <code>0</code>, program by m&#283;l vypsat obd&#233;ln&#237;k s &#353;&#237;&#345;kou <code>A</code> a v&#253;&#353;kou <code>B</code>.</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>0 3 5</code>:</p>
<pre><code>xxx
xxx
xxx
xxx
xxx</code></pre>
</li>
<li>
<p>(0.5b) Pokud m&#225; <code>obrazec</code> hodnotu <code>1</code>, vykreslete &#8220;dut&#253;&#8221; obd&#233;ln&#237;k s &#353;&#237;&#345;kou <code>A</code> a v&#253;&#353;kou <code>B</code>. Dut&#253; obd&#233;ln&#237;k obsahuje znaky <code>x</code> po sv&#233;m obvodu, ale uvnit&#345; je pr&#225;zdn&#253; (vypi&#353;te znak mezery).</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>1 4 5</code>:</p>
<pre><code>xxxx
x  x
x  x
x  x
xxxx</code></pre>
</li>
<li>
<p>(1b) Pokud m&#225; <code>obrazec</code> hodnotu <code>2</code>, vykreslete obd&#233;ln&#237;k s &#353;&#237;&#345;kou <code>A</code> a v&#253;&#353;kou <code>B</code>, kter&#253; bude m&#237;t uprost&#345;ed (mimo sv&#367;j obvod) &#269;&#237;sla se vzr&#367;staj&#237;c&#237; hodnotou. Hodnota &#269;&#237;sla bude za&#269;&#237;nat na nule, a bude v&#382;dy r&#367;st o jedni&#269;ku po &#345;&#225;dc&#237;ch (tj. nejprve sm&#283;rem doprava, pot&#233; sm&#283;rem dol&#367;). Vypisov&#233; &#269;&#237;slo nikdy nesm&#237; p&#345;es&#225;hnout hodnotu <code>9</code> (jinak by bylo moc &#353;irok&#233;), m&#237;sto toho by m&#283;lo &#8220;p&#345;et&#233;ct&#8221; zp&#283;t na za&#269;&#225;tek, tj. m&#237;sto <code>10</code> se vykresl&#237; <code>0</code>, m&#237;sto <code>11</code> se vykresl&#237; <code>1</code> atd.</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>2 6 7</code>:</p>
<pre><code>xxxxxx
x0123x
x4567x
x8901x
x2345x
x6789x
xxxxxx</code></pre>
</li>
<li>
<p>(0.5b) Pokud m&#225; <code>obrazec</code> hodnotu <code>3</code>, vykreslete diagon&#225;lu, kter&#225; bude m&#237;&#345;it doprava dol&#367;, a bude obsahovat <code>A</code> bod&#367; (hodnotu <code>B</code> zde ignorujte).</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>3 6 0</code>:</p>
<pre><code>x
 x
  x
   x
    x
     x</code></pre>
</li>
<li>
<p>(0.5b) Pokud m&#225; <code>obrazec</code> hodnotu <code>6</code>, vykreslete p&#237;smeno <code>T</code>, kter&#233; bude <code>A</code> bod&#367; &#353;irok&#233; a <code>B</code> bod&#367; vysok&#233;. M&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e hodnota <code>A</code> bude v&#382;dy lich&#225;.</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>6 5 4</code>:</p>
<pre><code>xxxxx
  x
  x
  x</code></pre>
</li>
<li>
<p>(Bonus 1b) Pokud m&#225; <code>obrazec</code> hodnotu <code>9</code>, vykreslete obd&#233;ln&#237;k s &#353;&#237;&#345;kou <code>A</code> a v&#253;&#353;kou <code>B</code>, kter&#253; bude m&#237;t uprost&#345;ed (mimo sv&#367;j obvod) &#269;&#237;sla se vzr&#367;staj&#237;c&#237; hodnotou. Narozd&#237;l od obrazce <code>2</code> se zde ale &#269;&#237;sla budou zvy&#353;ovat po sloupc&#237;ch, ne po &#345;&#225;dc&#237;ch.</p>
<p>Uk&#225;zkov&#253; v&#253;stup pro vstup <code>9 5 6</code>:</p>
<pre><code>xxxxx
x048x
x159x
x260x
x371x
xxxxx</code></pre>
</li>
<li><p>Pokud m&#225; <code>obrazec</code> jakoukoliv jinou hodnotu, vypi&#353;te &#345;&#225;dek <code>Neznamy obrazec</code> a ukon&#269;ete prorgam.</p></li>
</ul>
<h3 id="obecn&#233;-pozn&#225;mky">Obecn&#233; pozn&#225;mky</h3>
<ul>
<li>
<p>Pro tisk znak&#367; a od&#345;&#225;dkov&#225;n&#237; na v&#253;stup pou&#382;ijte <code>printf</code>:</p>
<pre class="c"><code>printf("x");  // v&#253;pis jednoho znaku
printf("\n"); // v&#253;pis od&#345;&#225;dkov&#225;n&#237;</code></pre>
</li>
<li><p>Pro v&#253;pis tvar&#367; pou&#382;ijte cykly a podm&#237;nky, smyslem &#250;lohy nen&#237; &#8220;natvrdo&#8221; vypsat konkr&#233;tn&#237; tvar, ale napsat k&#243;d, kter&#253; um&#237; tvar vypsat obecn&#283;, v z&#225;vislosti na hodnot&#283; vstupn&#237;ch parametr&#367; <code>A</code>/<code>B</code>.</p></li>
<li><p>M&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e vstup programu bude v&#382;dy v dan&#233;m form&#225;tu t&#345;&#237; &#269;&#237;sel odd&#283;len&#253;ch mezerou, krom&#283; neplatn&#233;ho &#269;&#237;sla obrazce (viz v&#253;&#353;e) nemus&#237;te v programu &#345;e&#353;it neplatn&#253; vstup.</p></li>
<li><p>M&#367;&#382;ete p&#345;edpokl&#225;dat, &#382;e v&#353;echna zadan&#225; &#269;&#237;sla budou nez&#225;porn&#225;.</p></li>
</ul>
<p>N&#225;pov&#283;da: p&#345;i programov&#225;n&#237; obrazc&#367; si uv&#283;domte, &#382;e na v&#253;stup programu um&#237;me kreslit pouze po &#345;&#225;dc&#237;ch. Mus&#237;te tedy v cyklu postupn&#283; proch&#225;zet jednotliv&#233; &#345;&#225;dky, v ka&#382;d&#233;m &#345;&#225;dku proch&#225;zet jednotliv&#233; sloupce, a v z&#225;vislosti na hodnot&#283; dan&#233;ho &#345;&#225;dku a sloupce vykreslit <code>x</code>, mezeru nebo &#269;&#237;slici.</p>
<h3 id="testy">Testy</h3>
<p>Po nahr&#225;n&#237; sv&#233;ho &#345;e&#353;en&#237; do Kelvina se spust&#237; automatick&#233; testy, kter&#233; zkontroluj&#237;, jestli V&#225;&#353; program vrac&#237; spr&#225;vn&#233; v&#253;stupy na odpov&#237;daj&#237;c&#237; vstupy. Obsahy test&#367; si m&#367;&#382;ete prohl&#233;dnout v z&#225;lo&#382;ce <a href="#tests">Tests</a>, kde si tak&#233; v&#353;echny testy m&#367;&#382;ete st&#225;hnout pomoc&#237; tla&#269;&#237;tka <code>Download all tests</code> vpravo naho&#345;e.</p>
</div>
