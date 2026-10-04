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
