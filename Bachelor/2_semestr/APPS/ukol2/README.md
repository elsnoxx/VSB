<p><strong>Tasks</strong></p>
<ol type="1">
<li>
<p>Najd&#283;te v poli &#269;&#237;sel <code>long</code> nejmen&#353;&#237; &#269;&#237;slo, kter&#233; m&#225; doln&#237; <strong>bajt</strong> nulov&#253;.</p>
<pre class="c"><code>long nejmensi_56bit( long *tp_array, int t_N );</code></pre>
</li>
<li>
<p>P&#345;eve&#271;te &#269;&#237;slo long na hex string.</p>
<pre class="c"><code>char g_tabulka[ 16 ] = "0123456789ABCDEF";

void long2hexstr( long t_num, char *tp_str ) {
  opakuj { tp_str[ i ] = g_tabulka[ t_num &amp; 0xF ]; t_num &gt;&gt;= 4; }
}</code></pre>
</li>
<li>
<p>Zjist&#283;te, zda je v &#345;et&#283;zci v&#237;ce mal&#253; &#269;i velk&#253;m p&#237;smen.</p>
<pre class="c"><code>int pismena( char *tp_str ) { if ( velke ) citac++; if ( male ) citac--; }</code></pre>
</li>
<li>
<p>Spo&#269;&#237;tejte faktori&#225;l &#269;&#237;sla <code>int</code> a v&#253;sledek vra&#357;te jako hodnotu <code>long</code>. Pokud dojde k p&#345;ete&#269;en&#237; p&#345;i v&#253;po&#269;tu, bude v&#253;sledek 0.</p>
<pre class="c"><code>long faktorial( int N );</code></pre>
</li>
<li>
<p>Kter&#233; &#269;&#237;slo v poli &#269;&#237;sel <code>int</code> m&#225; nejvy&#353;&#353;&#237; zbytek po d&#283;len&#237; &#269;&#237;slem <code>K</code>? Vynulujte v poli v&#353;echna &#269;&#237;sla, kter&#225; maj&#237; zbytek po d&#283;len&#237; men&#353;&#237;, ne&#382; ten nejvy&#353;&#353;&#237;.</p>
<pre class="c"><code>int nejvetsi_modulo( int *tp_pole, int t_N, int t_K );</code></pre>
</li>
<li><p>Implementujte si funkci pro p&#345;evod &#345;et&#283;zce na velk&#225; &#269;i mal&#225; p&#237;smena. Podm&#237;n&#283;n&#253; skok vyu&#382;ijte jen pro cyklus, pro p&#345;evod znak&#367; se sna&#382;te vyu&#382;&#237;t jen instrukce <code>CMOVxx</code>.</p></li>
<li><p>Ov&#283;&#345;te, zda je zadan&#233; &#269;&#237;slo <code>long</code> prvo&#269;&#237;slem.</p></li>
</ol>
</div>
