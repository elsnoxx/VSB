# Systém pro správu hotelu

## Popis projektu
Tento projekt je databázový systém navržený pro správu hotelových operací. Zahrnuje funkcionality pro správu hostů, zaměstnanců, typů pokojů, rezervací, plateb, služeb a zpětné vazby. Systém zajišťuje efektivní ukládání a vyhledávání dat pro hotelové řízení.

## Struktura databáze
Databáze se skládá z následujících tabulek:
1. **Guest**: Ukládá informace o hotelových hostech, včetně jejich osobních údajů a registračních dat.
2. **Employee**: Obsahuje podrobnosti o hotelových zaměstnancích a jejich pozicích.
3. **RoomType**: Definuje různé typy pokojů dostupných v hotelu.
4. **Room**: Představuje jednotlivé pokoje a jejich obsazenost.
5. **Payment**: Sleduje platební údaje pro rezervace a služby.
6. **Reservation**: Spravuje rezervace hostů, včetně dat check-in a check-out.
7. **Service**: Seznam služeb nabízených hotelem.
8. **ServiceUsage**: Zaznamenává využití služeb hosty během jejich pobytu.
9. **Feedback**: Shromažďuje zpětnou vazbu a hodnocení hostů za jejich pobyt.
10. **ServicePriceHistory**: Udržuje historii cen služeb v průběhu času.
11. **RoomTypePriceHistory**: Sleduje ceny typů pokojů v průběhu času.

## SQL koncepty a příklady

### GROUP BY CUBE
**Popis**: Umožňuje multidimenzionální analýzu dat. Generuje kombinace skupinování pro všechny možné úrovně atributů. CUBE vytváří všechny možné kombinace seskupení.

**Jak to funguje**: CUBE(city, guest_type) vytvoří následující skupiny:
- Všichni hosté (bez seskupení)
- Seskupení podle města
- Seskupení podle typu hosta
- Seskupení podle města a typu hosta současně

```sql
SELECT city, guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY CUBE(city, guest_type)
ORDER BY city, guest_type;
```

**Výsledek obsahuje**:
- Celkový počet hostů
- Počet hostů podle měst
- Počet hostů podle typů (standard, vip)
- Počet hostů podle kombinace město + typ

### Hierarchická data (Hierarchický výpis)
**Popis**: Používá hierarchické dotazování pomocí START WITH, CONNECT BY PRIOR, a LEVEL. Vhodné pro zobrazení stromové struktury organizace zaměstnanců.

**Jak to funguje**:
- START WITH definuje kořen hierarchie (zaměstnanci bez nadřízeného)
- CONNECT BY PRIOR definuje vztah mezi úrovněmi
- LEVEL ukazuje hloubku v hierarchii

```sql
SELECT firstname || ' ' || lastname AS "Employee Name", 
       LEVEL AS "Hierarchy Level",
       LPAD(' ', (LEVEL-1)*2) || firstname AS "Indented Name"
FROM Employee
START WITH manager_id IS NULL
CONNECT BY PRIOR employee_id = manager_id
ORDER BY employee_id;
```

### Množnicové operace
**Popis**: Operace umožňují kombinaci nebo porovnání výsledků z více dotazů.

#### UNION - Sjednocení
**Jak funguje**: Spojí výsledky bez duplicit
```sql
-- Všechna města kde bydlí hosté nebo zaměstnanci
SELECT city, 'Guest' AS source FROM Guest
UNION
SELECT city, 'Employee' AS source FROM Employee
ORDER BY city;
```

#### INTERSECT - Průnik
**Jak funguje**: Najde společné hodnoty mezi dotazy
```sql
-- Města kde bydlí současně hosté i zaměstnanci
SELECT city FROM Guest
INTERSECT
SELECT city FROM Employee;
```

#### MINUS - Rozdíl
**Jak funguje**: Hodnoty z prvního dotazu, které nejsou ve druhém
```sql
-- Města kde bydlí hosté, ale ne zaměstnanci
SELECT city FROM Guest
MINUS
SELECT city FROM Employee;
```

### Práce s daty
**Popis**: Manipulace s daty pomocí datových funkcí.

**Základní datové funkce**:
```sql
SELECT payment_date,
       TO_CHAR(payment_date, 'DD.MM.YYYY') AS formatted_date,
       ADD_MONTHS(payment_date, 3) AS payment_plus_3_months,
       LAST_DAY(payment_date) AS last_day_of_payment_month,
       MONTHS_BETWEEN(SYSDATE, payment_date) AS months_ago,
       SYSDATE AS current_date
FROM Payment
WHERE payment_date IS NOT NULL;
```

**Pokročilé práce s daty**:
```sql
-- Analýza plateb podle měsíců
SELECT TO_CHAR(payment_date, 'YYYY-MM') AS payment_month,
       COUNT(*) AS payment_count,
       SUM(total_accommodation) AS monthly_revenue
FROM Payment
WHERE payment_date IS NOT NULL
GROUP BY TO_CHAR(payment_date, 'YYYY-MM')
ORDER BY payment_month;
```

### FULL JOIN (FULL OUTER JOIN)
**Popis**: Spojuje dvě tabulky a zahrnuje všechny záznamy z obou tabulek, i když nemají odpovídající hodnoty.

**Jak funguje**: 
- Obsahuje všechny záznamy z levé tabulky
- Obsahuje všechny záznamy z pravé tabulky
- Pro neodpovídající záznamy doplní NULL hodnoty

```sql
-- Všichni hosté a jejich zpětná vazba (i hosté bez zpětné vazby a zpětné vazby bez hostů)
SELECT g.guest_id, g.firstname, g.lastname, 
       f.feedback_id, f.rating, f.note, f.feedback_date
FROM Guest g
FULL OUTER JOIN Feedback f ON g.guest_id = f.guest_id
ORDER BY g.lastname, f.feedback_date;
```

**Praktický příklad**:
```sql
-- Kompletní přehled rezervací a plateb
SELECT r.reservation_id, 
       g.firstname || ' ' || g.lastname AS guest_name,
       r.check_in_date, r.check_out_date, r.status,
       p.total_accommodation, p.is_paid, p.payment_date
FROM Reservation r
FULL OUTER JOIN Payment p ON r.payment_id = p.payment_id
JOIN Guest g ON r.guest_id = g.guest_id
ORDER BY r.check_in_date;
```

### Pokročilé funkce (FULL JOIN)
**Popis**: Využití FULL JOIN pro komplexní analýzy

```sql
-- Analýza všech služeb a jejich využití
SELECT s.service_id, s.name AS service_name,
       NVL(usage_stats.usage_count, 0) AS times_used,
       NVL(usage_stats.total_revenue, 0) AS total_revenue
FROM Service s
FULL OUTER JOIN (
    SELECT service_id, 
           COUNT(*) AS usage_count,
           SUM(total_price) AS total_revenue
    FROM ServiceUsage
    GROUP BY service_id
) usage_stats ON s.service_id = usage_stats.service_id
ORDER BY times_used DESC;
```

## Jak používat
1. Naplňte databázi pomocí dodaných SQL skriptů (`create.sql` a `data.sql`).
2. Používejte SQL dotazy pro interakci s databází při správě hotelových operací.
3. Odkazujte na výše uvedené příklady pro složité analýzy dat a reportování.

## Soubory
- `create.sql`: Obsahuje SQL příkazy pro vytvoření schématu databáze.
- `data.sql`: Zahrnuje ukázková data pro testování databáze.
- `SQL.sql`: Obsahuje kompletní příklady všech SQL konceptů a technik.
- `README.md`: Dokumentace projektu.

## Autor
Tento projekt byl vyvinut jako součást kurzu TDS1 ve 4. semestru.