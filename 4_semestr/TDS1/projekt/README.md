# Seznam praktických témat a příkladů v Oracle SQL

## 🧩 Základy SQL

### Řetězce: CONCAT, ||
- `CONCAT('Hello', 'World')` nebo `'Hello' || 'World'` - spojení textů
- Pozor: || má vyšší prioritu než CONCAT

### Porovnávání textů: LOWER, UPPER, INITCAP
- `LOWER('TEXT')` = 'text', `UPPER('text')` = 'TEXT'
- `INITCAP('hello world')` = 'Hello World' - každé slovo velkým písmenem

### Podmínky: WHERE, BETWEEN, LIKE, IN, IS NULL, NOT
- `WHERE age BETWEEN 18 AND 65` - včetně krajních hodnot
- `WHERE name LIKE 'J%'` - začíná na J, `%` = libovolný počet znaků, `_` = jeden znak
- `WHERE city IN ('Praha', 'Brno')` - hodnota v seznamu
- `IS NULL` vs `= NULL` - pouze IS NULL funguje správně!

### Logické operátory: AND, OR, NOT, závorkování
- Priorita: NOT > AND > OR
- Vždy používejte závorky pro přehlednost: `(A AND B) OR (C AND D)`

## 📊 Řazení, funkce a výpočty

### ORDER BY, ASC, DESC
- `ORDER BY salary DESC, last_name ASC` - více kritérií
- NULL hodnoty: Oracle defaultně řadí NULL jako největší hodnoty

### Agregační funkce: MIN, MAX, AVG, SUM, COUNT
- `COUNT(*)` vs `COUNT(column)` - COUNT(*) počítá i NULL hodnoty
- `AVG(salary)` ignoruje NULL hodnoty automaticky
- Používejte s GROUP BY pro seskupení

### Textové funkce: SUBSTR, LENGTH, INSTR, LPAD, TRIM, REPLACE
- `SUBSTR('Oracle', 2, 3)` = 'rac' - od pozice 2, délka 3
- `INSTR('Oracle', 'ac')` = 3 - pozice výskytu
- `LPAD('123', 5, '0')` = '00123' - doplnění zleva
- `TRIM(' text ')` = 'text' - odstranění mezer

### Číselné funkce: ROUND, TRUNC, MOD
- `ROUND(15.678, 2)` = 15.68, `ROUND(15.678, -1)` = 20
- `TRUNC(15.678, 1)` = 15.6 - bez zaokrouhlení
- `MOD(17, 5)` = 2 - zbytek po dělení

### Datumové funkce: SYSDATE, MONTHS_BETWEEN, ADD_MONTHS, LAST_DAY, NEXT_DAY
- `SYSDATE` - aktuální datum a čas
- `MONTHS_BETWEEN(SYSDATE, hire_date)` - rozdíl v měsících
- `ADD_MONTHS(SYSDATE, 3)` - přidání 3 měsíců
- `LAST_DAY(SYSDATE)` - poslední den v měsíci
- `NEXT_DAY(SYSDATE, 'MONDAY')` - nejbližší pondělí

## 🧮 Formátování a konverze

### TO_CHAR, TO_NUMBER, TO_DATE
- `TO_CHAR(SYSDATE, 'DD.MM.YYYY')` - formátování data
- `TO_CHAR(salary, '999,999.99')` - formátování čísla
- `TO_NUMBER('123.45')` - převod na číslo
- `TO_DATE('31.12.2023', 'DD.MM.YYYY')` - převod na datum

### NVL, NVL2, COALESCE, NULLIF
- `NVL(commission, 0)` - pokud NULL, pak 0
- `NVL2(commission, salary+commission, salary)` - různé hodnoty podle NULL/NOT NULL
- `COALESCE(val1, val2, val3)` - první nenulová hodnota
- `NULLIF(val1, val2)` - NULL pokud se rovnají, jinak val1

## 🔁 Podmínky a větvení

### DECODE, CASE, IF-THEN-ELSE

```sql
-- DECODE (starší syntax)
DECODE(status, 'A', 'Active', 'I', 'Inactive', 'Unknown')

-- CASE (preferovaný)
CASE 
  WHEN salary > 50000 THEN 'High'
  WHEN salary > 30000 THEN 'Medium'
  ELSE 'Low'
END
```

## 🔗 Spojování tabulek (JOIN)

### INNER, LEFT, RIGHT, FULL OUTER JOIN
- `INNER JOIN` - pouze záznamy s odpovídajícími hodnotami v obou tabulkách
- `LEFT JOIN` - všechny záznamy z levé tabulky + odpovídající z pravé
- `RIGHT JOIN` - všechny záznamy z pravé tabulky + odpovídající z levé
- `FULL OUTER JOIN` - všechny záznamy z obou tabulek

### NATURAL JOIN, CROSS JOIN
- `NATURAL JOIN` - automatické spojení podle stejně pojmenovaných sloupců
- `CROSS JOIN` - kartézský součin (každý s každým)

### JOIN ... USING, JOIN ... ON
- `JOIN table2 USING (column_name)` - pro stejně pojmenované sloupce
- `JOIN table2 ON table1.id = table2.table1_id` - explicitní podmínka

### Alternativní JOIN pomocí WHERE a +
- Oracle specifická syntax: `WHERE a.id = b.id(+)` = LEFT JOIN
- Nedoporučuje se, používejte standardní JOIN syntax

## 🧱 Hierarchie a rekurze

### START WITH, CONNECT BY PRIOR, LEVEL

```sql
-- Hierarchický dotaz
SELECT LEVEL, employee_id, manager_id, first_name
FROM employees
START WITH manager_id IS NULL  -- kořen stromu
CONNECT BY PRIOR employee_id = manager_id  -- podmínka hierarchie
ORDER SIBLINGS BY first_name;  -- řazení na stejné úrovni
```

## 📦 Agregace a seskupení

### GROUP BY, HAVING
- `GROUP BY` - seskupení dat pro agregační funkce
- `HAVING` - filtrování po seskupení (WHERE je před seskupením)

```sql
SELECT department_id, AVG(salary)
FROM employees
WHERE hire_date > '01.01.2020'
GROUP BY department_id
HAVING AVG(salary) > 50000;
```

### ROLLUP, CUBE, GROUPING SETS

#### ROLLUP
Vytváří subtotály a celkový součet. Výsledek obsahuje:
1. Subtotály pro každou kombinaci (department_id, job_id)
2. Subtotály pouze pro department_id (job_id = NULL)
3. Celkový součet (department_id = NULL, job_id = NULL)

```sql
SELECT department_id, job_id, SUM(salary)
FROM employees
GROUP BY ROLLUP(department_id, job_id);
```

#### CUBE
Všechny možné kombinace seskupení. Výsledek obsahuje:
1. Kombinace (department_id, job_id)
2. Pouze department_id (job_id = NULL)
3. Pouze job_id (department_id = NULL)
4. Celkový součet (oba = NULL)

```sql
SELECT department_id, job_id, SUM(salary)
FROM employees
GROUP BY CUBE(department_id, job_id);
```

#### GROUPING SETS
Explicitní definice skupin pro agregaci

## 🧰 Množinové operace

### UNION - Sjednocení bez duplikátů
- Spojuje výsledky dvou dotazů a automaticky odstraňuje duplikáty
- Výsledek je seřazen (Oracle interně třídí pro odstranění duplikátů)
- Pomalejší než UNION ALL kvůli kontrole duplikátů

```sql
-- Všichni zaměstnanci a manažeři (bez duplikátů)
SELECT employee_id, first_name, 'Employee' as type FROM employees
UNION
SELECT manager_id, first_name, 'Manager' as type FROM employees WHERE manager_id IS NOT NULL;
```

### UNION ALL - Sjednocení s duplikáty
- Spojuje výsledky bez kontroly duplikátů
- Rychlejší než UNION (nemusí třídit a kontrolovat)
- Zachovává původní pořadí záznamů
- Preferovaný způsob, pokud víte, že duplikáty nevzniknou

```sql
-- Všechny platby - za rezervace i služby
SELECT reservation_id as id, amount, 'Reservation' as payment_type FROM reservation_payments
UNION ALL
SELECT service_id as id, amount, 'Service' as payment_type FROM service_payments;
```

### INTERSECT - Průnik množin
- Vrací pouze záznamy, které existují v obou dotazech
- Automaticky odstraňuje duplikáty
- Užitečné pro nalezení společných prvků

```sql
-- Zaměstnanci, kteří jsou zároveň manažery
SELECT employee_id, first_name FROM employees
INTERSECT
SELECT manager_id, first_name FROM employees WHERE manager_id IS NOT NULL;
```

### MINUS - Rozdíl množin (A - B)
- Vrací záznamy z prvního dotazu, které NEJSOU ve druhém dotazu
- Automaticky odstraňuje duplikáty
- Pořadí dotazů je důležité: A MINUS B ≠ B MINUS A

```sql
-- Zaměstnanci, kteří NEJSOU manažery
SELECT employee_id, first_name FROM employees
MINUS
SELECT manager_id, first_name FROM employees WHERE manager_id IS NOT NULL;
```

### Důležitá pravidla pro množinové operace

#### Kompatibilita sloupců
- **Stejný počet sloupců** v obou dotazech
- **Kompatibilní datové typy** na stejných pozicích
- **Názvy sloupců** se berou z prvního dotazu

```sql
-- SPRÁVNĚ - stejný počet sloupců, kompatibilní typy
SELECT employee_id, salary FROM employees
UNION
SELECT department_id, budget FROM departments;

-- CHYBNĚ - různý počet sloupců
SELECT employee_id, first_name, salary FROM employees
UNION
SELECT department_id, name FROM departments;  -- chyba!
```

#### Praktické tipy
- **Používejte aliasy** pro lepší čitelnost výsledků
- **ORDER BY** lze použít pouze na konci celého výrazu
- **Závorky** můžete použít pro seskupení operací

```sql
-- Komplexní příklad s více operacemi
(SELECT 'Active' as status, COUNT(*) as count FROM employees WHERE status = 'A'
 UNION ALL
 SELECT 'Inactive' as status, COUNT(*) FROM employees WHERE status = 'I')
ORDER BY status;
```

#### Výkonnostní aspekty
- **UNION ALL** je rychlejší než **UNION** (žádné třídění)
- **INTERSECT** a **MINUS** mohou být pomalé u velkých tabulek
- Zvažte použití **JOIN** místo **INTERSECT** pro lepší výkon
- Indexy na sloupcích použitých v množinových operacích zlepšují výkon

```sql
-- Místo INTERSECT použijte EXISTS (často rychlejší)
SELECT e.employee_id, e.first_name 
FROM employees e
WHERE EXISTS (SELECT 1 FROM employees m WHERE m.manager_id = e.employee_id);
```

#### Alternativní přístupy
```sql
-- MINUS lze nahradit pomocí NOT EXISTS
SELECT e.employee_id, e.first_name 
FROM employees e
WHERE NOT EXISTS (SELECT 1 FROM employees m WHERE m.manager_id = e.employee_id);

-- UNION lze nahradit pomocí CASE
SELECT employee_id, 
       CASE WHEN condition1 THEN value1 
            WHEN condition2 THEN value2 
       END as result
FROM table_name;
```

## 🧠 Poddotazy

### Jednořádkové a víceřádkové
- Jednořádkové: `WHERE salary > (SELECT AVG(salary) FROM employees)`
- Víceřádkové: `WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = 'Praha')`

### IN, EXISTS, ANY, ALL
- `EXISTS` - testuje existenci záznamu (často rychlejší než IN)
- `ANY` - porovnání s libovolnou hodnotou ze seznamu
- `ALL` - porovnání se všemi hodnotami ze seznamu

### WITH ... AS (common table expressions)

```sql
WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 50000
)
SELECT department_id, COUNT(*) 
FROM high_earners 
GROUP BY department_id;
```

## 🧾 DML: Práce s daty

### INSERT, UPDATE, DELETE, MERGE
- `INSERT INTO table VALUES (...)` nebo `INSERT INTO table SELECT ...`
- `UPDATE table SET column = value WHERE condition`
- `DELETE FROM table WHERE condition`
- `MERGE` - kombinace INSERT/UPDATE/DELETE v jednom příkazu

### DEFAULT, SAVEPOINT, ROLLBACK, COMMIT
- `DEFAULT` - použití výchozí hodnoty sloupce
- `SAVEPOINT sp1` - vytvoření bodu návratu
- `ROLLBACK TO sp1` - návrat k bodu, `ROLLBACK` - zrušení celé transakce
- `COMMIT` - potvrzení změn

## 🏗️ DDL: Práce se strukturou

### CREATE, ALTER, DROP, RENAME, TRUNCATE
- `CREATE TABLE` - vytvoření tabulky
- `ALTER TABLE` - změna struktury (ADD/MODIFY/DROP COLUMN)
- `DROP TABLE` - smazání tabulky
- `RENAME` - přejmenování objektu
- `TRUNCATE` - rychlé smazání všech dat (vs DELETE)

### SEQUENCE, INDEX, CONSTRAINT, VIEW, SYNONYM
- `SEQUENCE` - automatické generování čísel
- `INDEX` - zrychlení vyhledávání
- `CONSTRAINT` - integritní omezení
- `VIEW` - virtuální tabulka
- `SYNONYM` - alias pro objekt

## 🔒 Omezení (Constraints)

### NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY
- `NOT NULL` - povinná hodnota
- `UNIQUE` - jedinečná hodnota
- `PRIMARY KEY` - kombinace NOT NULL + UNIQUE + identifikátor záznamu
- `FOREIGN KEY` - odkaz na jinou tabulku

### CHECK, ON DELETE CASCADE, ON DELETE SET NULL
- `CHECK (salary > 0)` - vlastní validační pravidlo
- `ON DELETE CASCADE` - smazání podřízených záznamů
- `ON DELETE SET NULL` - nastavení NULL při smazání nadřízeného

## 🧬 Datové typy a jejich použití

### CHAR, VARCHAR2, CLOB, NUMBER, BLOB
- `CHAR(n)` - pevná délka, doplněno mezerami
- `VARCHAR2(n)` - proměnná délka, max n znaků
- `CLOB` - velké textové objekty (až 4GB)
- `NUMBER(p,s)` - číslo s přesností p a s desetinnými místy
- `BLOB` - binární data

### TIMESTAMP, INTERVAL
- `TIMESTAMP` - datum a čas s mikrosekundami
- `INTERVAL` - časový interval (např. '2' YEAR, '30' DAY)

## 🔐 Oprávnění

### GRANT, REVOKE, USER_TAB_PRIVS, ALL_TAB_PRIVS
- `GRANT SELECT ON table TO user` - udělení oprávnění
- `REVOKE SELECT ON table FROM user` - odebrání oprávnění
- Systémové pohledy pro kontrolu oprávnění

## 📚 Regulární výrazy

### REGEXP_LIKE, REGEXP_REPLACE, REGEXP_INSTR, REGEXP_SUBSTR, REGEXP_COUNT
- `REGEXP_LIKE(text, pattern)` - testování vzoru
- `REGEXP_REPLACE(text, pattern, replacement)` - nahrazení podle vzoru
- Užitečné pro validaci emailů, telefonních čísel, složité textové operace

## 🔍 Zobrazení metadat

### USER_CONSTRAINTS, USER_TAB_COLUMNS, USER_TAB_COMMENTS, USER_UNUSED_COL_TABS
- Systémové pohledy pro získání informací o struktuře databáze
- `USER_*` - objekty aktuálního uživatele
- `ALL_*` - objekty dostupné uživateli
- `DBA_*` - všechny objekty (pouze pro DBA)

# Teorie z projektu TDS I – Hotel s vysvětlením

## Rozdíl mezi daty a informacemi
🔹 Data jsou neupravená fakta (např. jméno, e-mail), informace je výsledek zpracování dat, která mají smysl (např. průměrná délka pobytu hostů).

## Entity, instance, atributy a identifikátory
🔹 Entita je objekt v systému (např. Guest), instance je konkrétní záznam, atributy popisují vlastnosti entity a identifikátor ji jednoznačně určuje (např. guest_id).

## Relace v databázi – kardinalita a povinnost
🔹 Popisuje, kolik záznamů v jedné tabulce souvisí s druhou (např. 1:N) a zda je vztah povinný nebo volitelný.

## ER diagram dle konvencí
🔹 Grafické znázornění entit, atributů a vztahů mezi nimi, pomocí standardních symbolů.

## Maticový diagram relací
🔹 Tabulka, která ukazuje, jaké relace existují mezi tabulkami (např. Guest – Reservation = 1:N).

## Supertypy a subtypy
🔹 Supertyp (např. Guest) obsahuje společné atributy, subtypy (VIP, Regular) pak přidávají specifika.

## Business pravidla systému
🔹 Pravidla, která určují logiku a validitu systému (např. rezervace může vzniknout jen, pokud je pokoj volný).

## Přenositelné vs. nepřenositelné vazby
🔹 Přenositelné konstrukce fungují ve všech SŘBD (např. FOREIGN KEY), nepřenositelné jsou specifické pro Oracle (např. GENERATED ALWAYS AS IDENTITY).

## M:N relace s a bez informace
🔹 M:N relace může být prostá (jen propojení) nebo obsahovat další informace (např. množství služby ve ServiceUsage).

## 1:N identifikační relace
🔹 Pokud cizí klíč v potomkovi zároveň tvoří primární klíč, mluvíme o identifikačním vztahu (např. room_type_id v RoomTypePriceHistory).

## Normalizace (1NF, 2NF, 3NF)
🔹 Proces návrhu tabulek tak, aby nevznikala zbytečná redundance a neplatná data; v 3NF už nejsou žádné tranzitivní závislosti.

## ARC – Alternativní vztahové omezení
🔹 Říká, že záznam může mít vztah buď k jednomu objektu nebo k jinému, ale ne k oběma současně (např. platba je buď za rezervaci nebo službu).

## Hierarchické a rekurzivní relace
🔹 Vztah v rámci jedné tabulky – např. zaměstnanec může být nadřízeným jiného zaměstnance (manager_id odkazuje na employee_id).

## Historie dat
🔹 Změny v čase jsou ukládány do history tabulek (např. RoomTypePriceHistory obsahuje historii cen pokojů).

## Journaling
🔹 Podobné jako historie, ale sleduje změny stavu (např. platové změny zaměstnanců, změny pracovního zařazení).

## Úprava návrhu dle konvencí
🔹 Zajišťuje, aby ERD a názvy byly čitelné, srozumitelné a strukturované (např. diagram zaměřený na Reservation jako středobod).

## Generické modelování
🔹 Obecný model použitelný na více typů dat nebo projektů – zvyšuje znovupoužitelnost a přehlednost oproti klasickému návrhu.

## Integritní omezení
🔹 Pravidla pro zachování správnosti dat: PRIMARY KEY, FOREIGN KEY, NOT NULL, UNIQUE, CHECK, DEFAULT, atd.

## Relace mezi konceptuálním a relačním modelem
🔹 Proces převodu z ER diagramu do konkrétní struktury databázových tabulek a vysvětlení případných změn.

