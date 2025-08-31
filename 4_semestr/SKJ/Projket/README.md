# Zadání

<div class="tab-pane active" id="tab_assignment">
                <div>
<h2>Materiály:</h2>
<ul>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/template/tasks.py">tasks.py</a> - zadání</li>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/template/tests.py">tests.py</a> - testy</li>
<li>
<a href="/task/SKJ/2024S/GAU01/c08/asset/assets/skj_stack.pdf">skj_stack.pdf</a> - prezentace</li>
</ul>
<h2>Odevzdání</h2>
<p>Řešení úlohy odevzdávejte zde do Kelvinu (odevzdejte vyřešený soubor <code>tasks.py</code>).
Za každý správně naimplementovaný test dostanete 1 bod, tedy maximálně můžete získat 5 bodů.</p>
<p>V případě, že neodevzdáte do konce hodiny,
můžete úlohu dodělat doma a odevzdat do večera 23:59.</p>
<p>Podmínkou pro udělení bodů za vypracování doma je účast na hodině. Jen v případě, že se vám to nepodaří na hodině, tak můžete dodělat doma.</p>
<h2>Další informace</h2>
<p>Základní informace o předmětu naleznete <a href="https://github.com/geordi/skj-course">zde</a>.</p>
<h3>Nastavení prostředí a testů</h3>
<p>Při používání Pythonu si vždy vytvořte virtuální prostředí (zatím bude stačit jedno sdílené pro SKJ).
Nainstalujte si do něj knihovnu <code>pytest</code>, abyste mohli spouštět připravené unit testy.
</p>
<pre class="highlight"><code class="language-bash hljs" data-highlighted="yes">$ python3 -m venv venv_dir_path     <span class="hljs-comment"># Vytvoří virtuální prostředí pro instalaci balíčků (spusťte pouze jednou)</span>
$ <span class="hljs-built_in">source</span> venv_dir_path/bin/activate <span class="hljs-comment"># Aktivuje virtuální prostředí (spusťte po zapnutí terminálu)</span>
    <span class="hljs-comment"># na windows je cesta k activate scriptu: venv_dir_path/Scripts/Activate s příponou .ps1 nebo .bat dle konzole</span>
(venv) $ pip install pytest         <span class="hljs-comment"># Nainstaluje balíček pip do virtuálního prostředí (spusťte pouze jednou)</span>
(venv) $ python -m pytest tests.py  <span class="hljs-comment"># Spustí testy ze souboru tests.py</span>
(venv) $ python -m pytest -v tests.py   <span class="hljs-comment"># přepínač -v vypíše v jakých případech máte chybu</span>
(venv) $ python -m pytest -vv tests.py   <span class="hljs-comment"># více v, více verbose</span></code></pre>
</div>

</div>

<br><br><br>


# Hotelový rezervační systém

Aplikace je zaměřena na komplexní objednávkový systém pro hotely. Uživatelé mají možnost se zaregistrovat, přihlásit se do systému a následně si zarezervovat pokoj podle svých preferencí a požadavků. Po vytvoření rezervace může administrátor nebo zaměstnanec hotelu přidávat využité služby k danému pobytu (např. strava, wellness, parkování) a potvrdit jejich zaplacení. Zákazníci mají po dokončení pobytu a uhrazení platby možnost vyplnit zpětnou vazbu, ve které mohou ohodnotit kvalitu služeb a celkový dojem z pobytu. Pro administrátory systém nabízí kompletní administrativní rozhraní, které umožňuje efektivní správu hostů, pokojů, rezervací a dalších entit. Součástí aplikace je také REST API rozhraní, které poskytuje přístup k datům pro integraci s dalšími systémy nebo vytvoření mobilních aplikací.

## Splněné požadavky s popisem a umístěním v projektu

1. **Aplikace obsahuje alespoň 6 modelů, které jsou vzájemně provázány vazbou**
   - Modely: `Guest`, `Room`, `Reservation`, `Payment`, `RoomType`, `Service`, `ServiceUsage`
   - Najdeš v souboru [`blog/models.py`](blog/models.py)
   - Propojení je realizováno pomocí `ForeignKey`, `OneToOneField` a `ManyToManyField`.

2. **K modelům je vytvořeno administrativní rozhraní**
   - Django admin rozhraní pro všechny modely. Také implementované své vlastní rozhraní přístupné pro administrátora. Odkazy na administtraci se po přihlášení zobrazí v navigační liště.

3. **Aplikace obsahuje alespoň 12 view a odpovídající URL**
   - Například: seznam hostů, detail hosta, editace hosta, seznam rezervací, detail rezervace, editace rezervace, seznam pokojů, atd.
   - Vše v souboru [`blog/views.py`](blog/views.py)

4. **View předávají obsah templatům (12 templatů)**
   - Každé view má vlastní šablonu pro zobrazení dat.
   - Vše ve složce [`blog/templates/`](blog/templates/)
   - Například: `management/guest/guest_list.html`, `management/guest/guest_detail.html`, `management/reservation/reservation_list.html`, atd.

5. **Aplikace obsahuje alespoň 6 formulářů**
   - Použity Django ModelForm i vlastní formuláře (např. pro hosta, rezervaci, platbu, vyhledávání, přihlášení, registraci).
   - Vše v souboru [`blog/forms.py`](blog/forms.py)

6. **REST API k datům aplikace**
   - Implementováno pomocí Django Ninja (`blog/api.py`).
   - K dispozici jsou tyto endpointy:

| URL                       | Popis                                 |
|---------------------------|---------------------------------------|
| `/api/guests/`            | Seznam hostů (Guest)                  |
| `/api/roomtypes/`         | Seznam typů pokojů (RoomType)         |
| `/api/rooms/`             | Seznam pokojů (Room)                  |
| `/api/reservations/`      | Seznam rezervací (Reservation)        |
| `/api/services/`          | Seznam služeb (Service)               |
| `/api/payments/`          | Seznam plateb (Payment)               |
| `/api/serviceusages/`     | Seznam využití služeb (ServiceUsage)  |
| `/api/feedbacks/`         | Seznam hodnocení (Feedback)           |

   - Vše v [`blog/api.py`](blog/api.py)

7. **Aplikace tvoří logický celek a stránky jsou propojené odkazy**
   - Navigace mezi stránkami je zajištěna pomocí odkazů v šablonách.
   - Vše ve složce [`blog/templates/`](blog/templates/), např. v souborech `navbar.html`, `reservation_detail.html`, `guest_detail.html`.

8. **Aplikace není blog, fórum ani benzínová stanice**
   - Téma aplikace je hotelová evidence a rezervace pokoje.

9. **Aplikace obsahuje grafické prvky a CSS styly**
   - Pro vzhled je použit Bootstrap a Bootstrap Icons.
   - Vše v šablonách ve složce [`blog/templates/`](blog/templates/), např. v souborech `base.html`, `guest_form.html`, `reservation_list.html`.

10. **Aplikace neobsahuje geografickou složku**
    - Žádné mapy ani geolokace nejsou použity.

11. **Aplikace nepoužívá generické view**
    - Všechna view jsou implementována jako vlastní funkce.
    - Vše v [`blog/views.py`](blog/views.py)

---

## Spuštění aplikace v Dockeru

- Build:  
  `docker build -t djangoprojekt .`
- Spuštění:  
  `docker run -d -p 8000:8000 --name djangoprojekt_container djangoprojekt`
- Zastavení:  
  `docker stop djangoprojekt_container`
- Smazání:  
  `docker rm djangoprojekt_container`

---

**Admin přístup:**  
Uživatelské jméno: `admin`  
Heslo: `admin`

---

**Testovací uživatelé:**
- `elsnoxx` / `Benjamin1*`

---

Autor:  
Richard Ficek