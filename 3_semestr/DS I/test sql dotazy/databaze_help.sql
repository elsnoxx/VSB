WITH t as (
	-- TOTO VYPISE AUTORA, OBOR, POCET CLANKU V TOM OBORU OD AUTORA, A RANK JAKY JE TEN AUTOR V TOM DANEM OBORU
	SELECT 
		z_author.rid as author_id, 
		z_author.name as author_name, 
		z_field_of_science.sid as field_of_science_id,
		z_field_of_science.name as field_of_science_name, 
		COUNT(DISTINCT z_article.aid) as articles_in_category,
		DENSE_RANK() OVER(PARTITION BY z_field_of_science.sid ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
	FROM z_field_of_science 
	JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
	JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
	JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
	JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year
	JOIN z_article_author ON z_article.aid = z_article_author.aid
	RIGHT JOIN z_author ON z_article_author.rid = z_author.rid
	GROUP BY z_author.rid, z_author.name, 	
)

SELECT * FROM t
WHERE EXISTS (
	SELECT 1
	FROM t tt 
	WHERE tt.field_of_science_id = t.field_of_science_id 
	-- TADY JE TOP N PER GROUP (ZDE KONKRETNE TOP 3 AUTHORY S MAX CLANKY V DANEM FIELD OF SCINCE
	AND tt.rn IN (1,2,3)
	AND t.author_id = tt.author_id
)
ORDER BY t.field_of_science_id DESC, t.articles_in_category DESC



-- TOTO VYPISE POCET CLANKU PRO AUTORA V RAMCI FIELD FORD SCINCE
SELECT 
	z_author.rid as author_id, 
	z_author.name as author_name, 
	z_field_of_science.sid as field_of_science_id,
	z_field_of_science.name as field_of_science_name, 
	COUNT(DISTINCT z_article.aid) as articles_in_category,
	DENSE_RANK() OVER(PARTITION BY z_field_of_science.sid ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
FROM z_field_of_science 
JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year
JOIN z_article_author ON z_article.aid = z_article_author.aid
RIGHT JOIN z_author ON z_article_author.rid = z_author.rid
GROUP BY z_author.rid, z_author.name, z_field_of_science.sid, z_field_of_science.name

-- TOTO VYPISE POCET CLANKU PRO KAZDEHO AUTORA
SELECT 
	z_author.rid as author_id, 
	z_author.name as author_name, 
	COUNT(DISTINCT z_article.aid) as articles_in_category,
	DENSE_RANK() OVER(ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
FROM z_field_of_science 
JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year
JOIN z_article_author ON z_article.aid = z_article_author.aid
RIGHT JOIN z_author ON z_article_author.rid = z_author.rid
GROUP BY z_author.rid, z_author.name


-- TOTO VYPISE POCET CLANKU PRO KAZDOU INSTITUCI
(
SELECT 
	z_institution.iid as institution_id, 
	z_institution.name as institution_name, 
	COUNT(DISTINCT z_article.aid) as articles_in_category,
	DENSE_RANK() OVER(ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
FROM z_field_of_science 
JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year 
JOIN z_article_institution ON z_article.aid = z_article_institution.aid
RIGHT JOIN z_institution ON z_article_institution.iid = z_institution.iid
GROUP BY z_institution.iid, z_institution.name

)

-- TOTO VYPISE POCET CLANKU PRO KAZDOU INSTITUCI V RAMCI OBORU 
SELECT 
	z_institution.iid as institution_id, 
	z_institution.name as institution_name, 
	z_field_of_science.sid as field_of_science_id,
	z_field_of_science.name as field_of_science_name, 
	COUNT(DISTINCT z_article.aid) as articles_in_category,
	DENSE_RANK() OVER(PARTITION BY z_field_of_science.sid ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
FROM z_field_of_science 
JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year 
JOIN z_article_institution ON z_article.aid = z_article_institution.aid
RIGHT JOIN z_institution ON z_article_institution.iid = z_institution.iid
GROUP BY z_institution.iid, z_institution.name, z_field_of_science.sid, z_field_of_science.name



-- TOOT VYPISE TOP 2 INSTITUCE KTERE MAJI NEJVIC CLANKU ZA OBOR
WITH t as (
SELECT 
	z_institution.iid as institution_id, 
	z_institution.name as institution_name, 
	z_field_of_science.sid as field_of_science_id,
	z_field_of_science.name as field_of_science_name, 
	COUNT(DISTINCT z_article.aid) as articles_in_category,
	DENSE_RANK() OVER(PARTITION BY z_field_of_science.sid ORDER BY COUNT(DISTINCT z_article.aid) DESC) as rn
FROM z_field_of_science 
JOIN z_field_ford ON z_field_of_science.sid = z_field_ford.sid
JOIN z_year_field_journal ON z_field_ford.fid = z_year_field_journal.fid
JOIN z_journal ON z_year_field_journal.jid = z_journal.jid
JOIN z_article ON z_journal.jid = z_article.jid AND z_year_field_journal.year = z_article.year 
JOIN z_article_institution ON z_article.aid = z_article_institution.aid
RIGHT JOIN z_institution ON z_article_institution.iid = z_institution.iid
GROUP BY z_institution.iid, z_institution.name, z_field_of_science.sid, z_field_of_science.name
)

SELECT * FROM t
WHERE EXISTS (
	SELECT 1
	FROM t tt 
	WHERE tt.field_of_science_id = t.field_of_science_id 
	-- TADY JE TOP N PER GROUP (ZDE KONKRETNE TOP 3 AUTHORY S MAX CLANKY V DANEM FIELD OF SCINCE
	AND tt.rn IN (1,2)
	AND t.institution_id = tt.institution_id
)
ORDER BY t.field_of_science_id DESC


-- Dotazy v tomto souboru maji za ukol upozornit na problematicke casti databaze


-- Autori, kteri publikovali v casopise `adaptive behavior` v roce 2018.
-- Zkuste se zamyslet, proc v SQL dotazu musi byt distinct.
select distinct au.name, a.name, y.year, j.name
from z_journal j
join z_year_field_journal y on j.jid = y.jid
join z_article a on a.jid = y.jid and a.year = y.year
join z_article_author aa on a.aid = aa.aid
join z_author au on au.rid = aa.rid
where issn = '1059-7123' and a.year = 2018




-- Pocet clanku pro jednotlive obory FORD ve vedni oblasti 'Natural sciences'.
-- Do dotazu jsme pridali i COUNT(distinct a.aid), aby jsme nazorne videli, 
-- ze klausule DISTINCT neni v dotazu potreba.
select ff.name, COUNT(*), COUNT(distinct a.aid)
from z_field_ford ff
join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where fs.name = 'Natural sciences'
group by ff.fid, ff.name


-- Pocet clanku pro jednotlive obory FORD ve vedni oblasti 'Natural sciences'.
-- Do dotazu jsme pridali i COUNT(distinct a.aid), aby jsme nazorne videli, 
-- ze klausule DISTINCT neni v dotazu potreba.
select ff.name, COUNT(*), COUNT(distinct a.aid)
from z_field_ford ff
join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where fs.name = 'Natural sciences'
group by ff.fid, ff.name


-- Pocet clanku pro jednotlive instituce, ktere jsou z Prahy (tzn. ze jejich town zacina na Praha)
-- ve vedni oblasti 'Natural sciences'.
-- Do dotazu jsme pridali i COUNT(distinct a.aid), aby jsme nazorne videli, 
-- ze klausule DISTINCT je tentokrat v dotazu potreba. 
-- Toto je zpusobeno skutecnosti, ze casopis muze byt ve vice oborech.
select i.iid, i.name, COUNT(*), COUNT(distinct a.aid)
from z_field_ford ff
join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
join z_article_institution ai on a.aid = ai.aid
join z_institution i on i.iid = ai.iid
where i.town LIKE 'Praha%' and fs.name = 'Natural sciences'
group by i.iid, i.name




-- Vypiste pocet clanku v prvnim decilu pro kazdou instituci z Prahy 6.
select i.iid, i.name, COUNT(distinct a.aid)
from z_article a 
join z_year_field_journal j on a.jid = j.jid and a.year = j.year and j.ranking = 'Decil'
right join z_article_institution ai on a.aid = ai.aid
right join z_institution i on i.iid = ai.iid
where i.town LIKE 'Praha 6%'
group by i.iid, i.name

select i.iid, 
	i.name, 
	(
	   select COUNT(distinct a.aid)
	   from z_article_institution ai 
	   join z_article a on a.aid = ai.aid
	   join z_year_field_journal j on a.jid = j.jid and a.year = j.year
	   where ai.iid = i.iid and j.ranking = 'Decil'
	) clanky_v_prvnim_decilu
from z_institution i
where i.town LIKE 'Praha 6%'
