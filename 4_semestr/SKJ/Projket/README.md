# Projekt – Webová aplikace v Django

**Termín odevzdání:** na cvičeních v zápočtovém týdnu od 13.5.2024 do 17.5.2024.

## Splněné požadavky s popisem a umístěním v projektu

1. **Aplikace obsahuje alespoň 6 modelů, které jsou vzájemně provázány vazbou**
   - Modely: `Guest`, `Room`, `Reservation`, `Payment`, `RoomType`, `Expense`
   - Najdeš v souboru [`blog/models.py`](blog/models.py)
   - Propojení je realizováno pomocí `ForeignKey`, `OneToOneField` a `ManyToManyField`.

2. **K modelům je vytvořeno administrativní rozhraní**
   - Django admin rozhraní pro všechny modely.
   - Najdeš v souboru [`blog/admin.py`](blog/admin.py)
   - Přístup přes `/admin` po spuštění aplikace.

3. **Aplikace obsahuje alespoň 12 view a odpovídající URL**
   - Například: seznam hostů, detail hosta, editace hosta, seznam rezervací, detail rezervace, editace rezervace, seznam pokojů, atd.
   - Najdeš v souboru [`blog/views.py`](blog/views.py)
   - Odpovídající URL jsou v [`mysite/urls.py`](mysite/urls.py) a případně v `blog/urls.py` (pokud existuje).

4. **View předávají obsah templatům (12 templatů)**
   - Každé view má vlastní šablonu pro zobrazení dat.
   - Najdeš ve složce [`blog/templates/`](blog/templates/)
   - Například: `management/guest/guest_list.html`, `management/guest/guest_detail.html`, `management/reservation/reservation_list.html`, atd.

5. **Aplikace obsahuje alespoň 6 formulářů**
   - Použity Django ModelForm i vlastní formuláře (např. pro hosta, rezervaci, platbu, vyhledávání, přihlášení, registraci).
   - Najdeš v souboru [`blog/forms.py`](blog/forms.py)

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
   - Najdeš ve složce [`blog/templates/`](blog/templates/), např. v souborech `navbar.html`, `reservation_detail.html`, `guest_detail.html`.

8. **Aplikace není blog, fórum ani benzínová stanice**
   - Téma aplikace je hotelová evidence a rezervace pokoje.

9. **Aplikace obsahuje grafické prvky a CSS styly**
   - Pro vzhled je použit Bootstrap a Bootstrap Icons.
   - Najdeš v šablonách ve složce [`blog/templates/`](blog/templates/), např. v souborech `base.html`, `guest_form.html`, `reservation_list.html`.

10. **Aplikace neobsahuje geografickou složku**
    - Žádné mapy ani geolokace nejsou použity.

11. **Aplikace nepoužívá generické view**
    - Všechna view jsou implementována jako vlastní funkce (function-based views).
    - Najdeš v [`blog/views.py`](blog/views.py)

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
Uživatelské jméno: `sa`  
Heslo: `AdminTest`

---

**Testovací uživatelé:**
- `elsnoxx` / `Benjamin1*`
- `er` / `!`

---

Autor:  
elsnoxx