# Klíčové aspekty databázového modelu a SQL dotazů

Vzhledem k poskytnutým SQL dotazům a datovému modelu databáze z přiloženého souboru bych tě chtěl upozornit na několik klíčových aspektů, které mohou být problematické:

## 1. Potřeba DISTINCT v dotazech:
- V prvním dotazu se používá `SELECT DISTINCT` kvůli více vazbám mezi tabulkami (jako jsou vztahy mezi autory a články). Může se stát, že více záznamů v tabulce `z_article_author` povede k tomu, že jeden článek bude uveden několikrát, pokud má více autorů. Bez `DISTINCT` by se mohly zobrazit duplicitní řádky.
- Ve druhém dotazu je příklad toho, že `DISTINCT` není vždy nutné. Zde by použití `DISTINCT` v počítání článků (`COUNT`) nedávalo smysl, protože každý článek by měl být počítán pouze jednou na základě jedinečného ID článku (`a.aid`).

```sql
SELECT DISTINCT au.name, a.name, y.year, j.name
FROM z_journal j
JOIN z_year_field_journal y ON j.jid = y.jid
JOIN z_article a ON a.jid = y.jid AND a.year = y.year
JOIN z_article_author aa ON a.aid = aa.aid
JOIN z_author au ON au.rid = aa.rid
WHERE j.issn = '1059-7123' AND a.year = 2018;
```

## 2. Vazby mezi tabulkami:
- Tabulky `Year_Field_Journal` a `Article` musí být spojeny přes dva atributy (`jid` a `year`). Toto je důležité, protože časopis může být v daném roce přiřazen více oborům a mít různé hodnocení. Pokud bys zapomněl na toto propojení, mohl bys získat nesprávná nebo neúplná data.

```sql
SELECT ff.name, COUNT(*), COUNT(DISTINCT a.aid)
FROM z_field_ford ff
JOIN z_field_of_science fs ON ff.sid = fs.sid
JOIN z_year_field_journal yfj ON ff.fid = yfj.fid
JOIN z_article a ON a.jid = yfj.jid AND a.year = yfj.year
WHERE fs.name = 'Natural sciences'
GROUP BY ff.fid, ff.name;

```

## 3. Vícenásobné přiřazení článku do více oborů:
- V dotazu o počtu článků publikovaných institucemi z Prahy je uvedeno, že časopis může být přiřazen do více vědních oborů. Proto je použití `DISTINCT` v tomto případě potřeba, jinak by každý článek, který je přiřazen k více oborům, byl započítán několikrát.

```sql
SELECT i.iid, i.name, COUNT(*), COUNT(DISTINCT a.aid)
FROM z_field_ford ff
JOIN z_field_of_science fs ON ff.sid = fs.sid
JOIN z_year_field_journal yfj ON ff.fid = yfj.fid
JOIN z_article a ON a.jid = yfj.jid AND a.year = yfj.year
JOIN z_article_institution ai ON a.aid = ai.aid
JOIN z_institution i ON i.iid = ai.iid
WHERE i.town LIKE 'Praha%' AND fs.name = 'Natural sciences'
GROUP BY i.iid, i.name;

```

## 4. Práce s tabulkou `z_year_field_journal`:
- Tato tabulka obsahuje více hodnot pro stejný časopis a rok, pokud je zařazen do více oborů a kategorií hodnocení. Bez správného zpracování těchto vztahů bys mohl získat chybné výsledky, například duplicitní záznamy článků v různých kategoriích.

```sql
SELECT i.iid, i.name, COUNT(DISTINCT a.aid)
FROM z_article a 
JOIN z_year_field_journal j ON a.jid = j.jid AND a.year = j.year AND j.ranking = 'Decil'
RIGHT JOIN z_article_institution ai ON a.aid = ai.aid
RIGHT JOIN z_institution i ON i.iid = ai.iid
WHERE i.town LIKE 'Praha 6%'
GROUP BY i.iid, i.name;

```

## 5. Riziko chyb kvůli nedostatku identifikátorů osob:
- V souboru se zmiňuje, že databáze neobsahuje žádné identifikátory osob (autoři se rozlišují jen podle jména). To znamená, že různé osoby se stejným jménem jsou v databázi považovány za jednu osobu. To může vést k chybným výsledkům, pokud dotazujete informace o autorech.

```sql
SELECT i.iid, 
       i.name, 
       (
           SELECT COUNT(DISTINCT a.aid)
           FROM z_article_institution ai 
           JOIN z_article a ON a.aid = ai.aid
           JOIN z_year_field_journal j ON a.jid = j.jid AND a.year = j.year
           WHERE ai.iid = i.iid AND j.ranking = 'Decil'
       ) clanky_v_prvnim_decilu
FROM z_institution i
WHERE i.town LIKE 'Praha 6%';
```

**Pozor na tyto body ti pomůže zajistit správnou interpretaci výsledků a zabránit duplicitním či nesprávným datům.**


# SQL Dotazy pro databázový model

## 1. Články v prestižních časopisech dle oboru:

 Vypište názvy článků, které byly publikovány v časopisech hodnocených jako 'Q1' (první kvartil) v roce 2020 v oblasti Physics and Astronomy (obor dle klasifikace FORD). Zahrňte také název časopisu, ve kterém článek vyšel, a počet autorů tohoto článku.

```sql
SELECT a.name AS article_name, 
       j.name AS journal_name, 
       COUNT(aa.rid) AS author_count
FROM z_article a
JOIN z_year_field_journal yfj ON a.jid = yfj.jid AND a.year = yfj.year
JOIN z_journal j ON a.jid = j.jid
JOIN z_field_ford ff ON yfj.fid = ff.fid
JOIN z_article_author aa ON a.aid = aa.aid
WHERE yfj.ranking = 'Q1' 
  AND a.year = 2020 
  AND ff.name = 'Physics and Astronomy'
GROUP BY a.name, j.name;
```

## 2. Spolupráce autorů z různých institucí:

Vypište jména autorů, kteří v roce 2019 publikovali články, na kterých spolupracovali s alespoň třemi různými institucemi. U každého autora vypište počet těchto článků a názvy těchto institucí.

```sql
SELECT au.name AS author_name, 
       COUNT(DISTINCT a.aid) AS article_count, 
       GROUP_CONCAT(DISTINCT i.name) AS institution_names
FROM z_author au
JOIN z_article_author aa ON au.rid = aa.rid
JOIN z_article a ON aa.aid = a.aid
JOIN z_article_institution ai ON a.aid = ai.aid
JOIN z_institution i ON ai.iid = i.iid
WHERE a.year = 2019
GROUP BY au.name
HAVING COUNT(DISTINCT i.iid) >= 3;

```

## 3. Instituce a články v konkrétních vědních oborech:

Pro každou instituci z Brna vypište počet článků, které její autoři publikovali ve vědní oblasti 'Engineering' mezi lety 2018 a 2021. Zahrňte také názvy těchto článků a jejich rok publikace.

```sql
SELECT i.name AS institution_name, 
       a.name AS article_name, 
       a.year
FROM z_institution i
JOIN z_article_institution ai ON i.iid = ai.iid
JOIN z_article a ON ai.aid = a.aid
JOIN z_year_field_journal yfj ON a.jid = yfj.jid AND a.year = yfj.year
JOIN z_field_ford ff ON yfj.fid = ff.fid
WHERE i.town LIKE 'Brno%' 
  AND ff.name = 'Engineering'
  AND a.year BETWEEN 2018 AND 2021;

```

## 4. Nejaktivnější autoři v konkrétních časopisech:

Najděte pět autorů, kteří publikovali nejvíce článků v časopise Journal of Applied Mathematics mezi lety 2017 a 2020. U každého autora vypište jeho jméno, počet článků a názvy všech těchto článků.

```sql
SELECT au.name AS author_name, 
       COUNT(a.aid) AS article_count, 
       GROUP_CONCAT(a.name) AS article_names
FROM z_author au
JOIN z_article_author aa ON au.rid = aa.rid
JOIN z_article a ON aa.aid = a.aid
JOIN z_journal j ON a.jid = j.jid
WHERE j.name = 'Journal of Applied Mathematics'
  AND a.year BETWEEN 2017 AND 2020
GROUP BY au.name
ORDER BY article_count DESC
LIMIT 5;

```

## 5. Vývoj publikací v konkrétním oboru:

Vypište počet článků publikovaných každý rok mezi 2015 a 2020 ve vědní oblasti Computer Science (dle klasifikace FORD). Zahrňte také název časopisu, ve kterém byly tyto články publikovány, a celkový počet institucí, které na článcích spolupracovaly každý rok.

```sql
SELECT a.year, 
       COUNT(DISTINCT a.aid) AS article_count, 
       j.name AS journal_name, 
       COUNT(DISTINCT ai.iid) AS institution_count
FROM z_article a
JOIN z_year_field_journal yfj ON a.jid = yfj.jid AND a.year = yfj.year
JOIN z_field_ford ff ON yfj.fid = ff.fid
JOIN z_journal j ON a.jid = j.jid
JOIN z_article_institution ai ON a.aid = ai.aid
WHERE ff.name = 'Computer Science' 
  AND a.year BETWEEN 2015 AND 2020
GROUP BY a.year, j.name;

```

## 6. Autorské spolupráce mezi městy:

Vypište články publikované v roce 2020, na kterých spolupracovali autoři z institucí ve dvou různých městech. U každého článku vypište jeho název, názvy měst a počet autorů z každého města.

```sql
SELECT a.name AS article_name, 
       GROUP_CONCAT(DISTINCT i.town) AS cities, 
       COUNT(DISTINCT aa.rid) AS author_count
FROM z_article a
JOIN z_article_institution ai ON a.aid = ai.aid
JOIN z_institution i ON ai.iid = i.iid
JOIN z_article_author aa ON a.aid = aa.aid
WHERE a.year = 2020
GROUP BY a.name
HAVING COUNT(DISTINCT i.town) >= 2;

```

## 7. Počet publikací dle typu článku:

Vypište pro každý typ článku (Article.type) celkový počet článků publikovaných v roce 2021. Zahrňte také název časopisu, ve kterém byly tyto články publikovány, a počet různých institucí, které na nich spolupracovaly.

```sql
SELECT a.type AS article_type, 
       COUNT(DISTINCT a.aid) AS article_count, 
       j.name AS journal_name, 
       COUNT(DISTINCT ai.iid) AS institution_count
FROM z_article a
JOIN z_journal j ON a.jid = j.jid
JOIN z_article_institution ai ON a.aid = ai.aid
WHERE a.year = 2021
GROUP BY a.type, j.name;

```

## 8. Autorská produktivita v časopisech:

Vypište jména autorů, kteří v roce 2020 publikovali alespoň 3 články v různých časopisech. Pro každý článek vypište název článku, název časopisu a rok publikace.



```sql
SELECT au.name AS author_name, 
       a.name AS article_name, 
       j.name AS journal_name, 
       a.year
FROM z_author au
JOIN z_article_author aa ON au.rid = aa.rid
JOIN z_article a ON aa.aid = a.aid
JOIN z_journal j ON a.jid = j.jid
WHERE a.year = 2020
GROUP BY au.name, a.name, j.name, a.year
HAVING COUNT(DISTINCT a.jid) >= 3;

```

## 9. Instituce a články v prestižních časopisech:

Pro každou českou instituci vypište následující údaje: Počet článků, které její vědci publikovali v časopisech hodnocených v prvním decilu (Year_Field_Journal.ranking = 'Decil'). Počet různých vědních oborů, ve kterých tyto články vyšly. Název instituce a její ID. Zahrňte pouze instituce z Prahy.

```sql
SELECT i.name AS institution_name, 
       i.iid AS institution_id, 
       COUNT(DISTINCT a.aid) AS article_count, 
       COUNT(DISTINCT ff.fid) AS field_count
FROM z_institution i
JOIN z_article_institution ai ON i.iid = ai.iid
JOIN z_article a ON ai.aid = a.aid
JOIN z_year_field_journal yfj ON a.jid = yfj.jid AND a.year = yfj.year
JOIN z_field_ford ff ON yfj.fid = ff.fid
WHERE i.town LIKE 'Praha%' 
  AND yfj.ranking = 'Decil'
GROUP BY i.name, i.iid;

```
## 



```sql

```
## 



```sql

```
