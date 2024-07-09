<div>
<p>Va&#353;&#237;m &#250;kolem prvn&#237;ho t&#253;dne je zejm&#233;na nastavit si <a href="https://mrlvsb.github.io/upr-skripta/prostredi/nastaveni_prostredi.html">prost&#345;ed&#237;</a> pro v&#253;voj program&#367; pomoc&#237; jazyka <em>C</em>.</p>
<p>Zejm&#233;na byste si m&#283;li zprovoznit opera&#269;n&#237; syst&#233;m Linux, a&#357; u&#382; nativn&#283;, ve virtu&#225;ln&#237;m stroji nebo pod Windows pomoc&#237; WSL. Tak&#233; budete pot&#345;ebovat m&#237;t funk&#269;n&#237; p&#345;eklada&#269; k&#243;du a editor zdrojov&#233;ho k&#243;du, ide&#225;ln&#283; <a href="https://code.visualstudio.com/">VS Code</a>.</p>
<p>Co se t&#253;&#269;e programov&#225;n&#237;, tak si zkus&#237;te vytvo&#345;it a spustit ten nejz&#225;kladn&#283;j&#353;&#237; <em>C</em> program (tzv. &#8220;hello world&#8221;), dozv&#283;d&#283;t se o prom&#283;nn&#253;ch, v&#253;razech a datov&#253;ch typech. Velmi u&#382;ite&#269;n&#233; je tak&#233; sezn&#225;mit se s <a href="https://mrlvsb.github.io/upr-skripta/prostredi/ladeni.html">lad&#283;n&#237;m a krokov&#225;n&#237;m</a> k&#243;du.</p>
<p>Ve&#353;ker&#233; informace o u&#269;ivu p&#345;edm&#283;tu naleznete v online <a href="https://mrlvsb.github.io/upr-skripta">skriptech</a>.</p>
<p><strong>Pokud byste si s &#269;&#237;mkoliv nev&#283;d&#283;li rady, nev&#225;hejte kdykoliv napsat na <a href="https://discord-fei.vsb.cz/">&#353;koln&#237; Discord</a> do m&#237;stnosti pro <a href="https://discord.com/channels/631124326522945546/1058360071567978496/threads/1058362395896062042">UPR</a>.</strong></p>
<h2 id="kapitoly-ve-skriptech-k-t&#233;to-lekci">Kapitoly ve skriptech k t&#233;to lekci</h2>
<ul>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/prostredi/nastaveni_prostredi.html">Nastaven&#237; prost&#345;ed&#237;</a>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/prostredi/linux/linux.html">Linux</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/prostredi/editor.html">Editor k&#243;du</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/prostredi/preklad_programu.html">P&#345;eklad program&#367;</a></li>
</ul>
</li>
<li>
<a href="https://mrlvsb.github.io/upr-skripta/c/programovani.html">Programov&#225;n&#237; v <em>C</em></a>
<ul>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/syntaxe.html">Z&#225;klady syntaxe <em>C</em></a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/prikazy_vyrazy.html">P&#345;&#237;kazy a v&#253;razy</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/promenne/promenne.html">Prom&#283;nn&#233;</a></li>
<li><a href="https://mrlvsb.github.io/upr-skripta/c/datove_typy/datove_typy.html">Datov&#233; typy</a></li>
</ul>
</li>
</ul>
<p>&#218;lohy k procvi&#269;en&#237; naleznete <a href="https://mrlvsb.github.io/upr-skripta/ulohy/zaklady.html">zde</a>.</p>
<h2 id="dom&#225;c&#237;-&#250;loha">Dom&#225;c&#237; &#250;loha</h2>
<p><strong>Pokud detekujeme plagiarismus (opisov&#225;n&#237; odevzdan&#233; &#250;lohy), tak v&#353;em student&#367;m s duplikovan&#253;m nebo velmi podobn&#253;m &#345;e&#353;en&#237;m bude ud&#283;leno -10 bod&#367; (p&#345;i prvn&#237;m proh&#345;e&#353;ku), p&#345;&#237;padn&#283; -100 bod&#367; (p&#345;i druh&#233;m proh&#345;e&#353;ku).</strong></p>
<p>Vytvo&#345;te program, kter&#253; bude &#8220;rozm&#283;&#328;ovat&#8221; pen&#237;ze. Na za&#269;&#225;tku programu (ve funkci <code>main</code>) si vytvo&#345;te celo&#269;&#237;selnou <a href="https://mrlvsb.github.io/upr-skripta/c/promenne/promenne.html">prom&#283;nnou</a>, do kter&#233; ulo&#382;te hodnotu (&#269;&#237;slo), kter&#233; bude m&#237;t v&#225;&#353; program za &#250;kol rozm&#283;nit na jednotliv&#233; bankovky.</p>
<p>Program pot&#233; spo&#269;&#237;t&#225;, jak&#253; je nejmen&#353;&#237; po&#269;et &#269;esk&#253;ch bankovek pro rozm&#283;n&#283;n&#237; dan&#233; &#269;&#225;stky, a pot&#233; <a href="https://mrlvsb.github.io/upr-skripta/c/prikazy_vyrazy.html#v%C3%BDpis-v%C3%BDraz%C5%AF">vyp&#237;&#353;e</a> pot&#345;ebn&#253; po&#269;et pro ka&#382;d&#253; typ bankovky (<code>5000</code>, <code>2000</code>, <code>1000</code> atd.). U rozm&#283;&#328;ovan&#233; hodnoty ignorujte jednotky a des&#237;tky korun. Jinak &#345;e&#269;eno, po&#269;&#237;tejte s danou &#269;&#225;stkou, jako by byla zaokrouhlen&#225; dol&#367; na stovky.</p>
<p>N&#225;pov&#283;da: pod&#237;vejte se do <a href="https://mrlvsb.github.io/upr-skripta/c/datove_typy/celociselne_typy.html#tabulka-aritmetick%C3%BDch-oper%C3%A1tor%C5%AF">tabulky</a> aritmetick&#253;ch oper&#225;tor&#367; jazyka <em>C</em> a zamyslete se nad t&#237;m, kter&#233; operace by se v&#225;m pro tento v&#253;po&#269;et mohly hodit. Jakou matematickou operac&#237; zjist&#237;te, kolik bankovek pot&#345;ebuji na dosa&#382;en&#237; n&#283;jak&#233; &#269;&#225;stky, a jakou operaci pot&#345;ebujete na zji&#353;t&#283;n&#237;, kolik pot&#233; z &#269;&#225;stky zbude?</p>
<p>P&#345;&#237;klad v&#253;stupu programu pro &#269;&#225;stku <code>9420</code>:</p>
<pre><code>Bankovka 5000: 1x
Bankovka 2000: 2x
Bankovka 1000: 0x
Bankovka 500: 0x
Bankovka 200: 2x
Bankovka 100: 0x</code></pre>
<p>P&#345;&#237;klad v&#253;stupu programu pro &#269;&#225;stku <code>8600</code>:</p>
<pre><code>Bankovka 5000: 1x
Bankovka 2000: 1x
Bankovka 1000: 1x
Bankovka 500: 1x
Bankovka 200: 0x
Bankovka 100: 1x</code></pre>
<p>P&#345;&#237;klad v&#253;stupu programu pro &#269;&#225;stku <code>250</code>:</p>
<pre><code>Bankovka 5000: 0x
Bankovka 2000: 0x
Bankovka 1000: 0x
Bankovka 500: 0x
Bankovka 200: 1x
Bankovka 100: 0x</code></pre>
</div>
