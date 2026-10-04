-- nwm 
select distinct au.name, a.name, y.year, j .name
from z_journal j
join z_year_field_journal y on j.jid = y.jid
--join z_field_ford ff on ff.fid = y.fid
join z_article a on a.jid = y.jid and a.year = y.year
join z_article_author aa on aa.aid = a.aid
join z_author au on au.rid = aa.rid
where issn = '1059-7123' and a.year = 2018

-- kolik clanku ma obor = 12
select count(*)
from z_field_ford ff
join z_year_field_journal yfj on yfj.fid = ff.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where ff.fid = 12

-- problem byl ze muze byt clanek prirazen k vice oborum
select count(*)
from z_field_ford ff
join z_year_field_journal yfj on yfj.fid = ff.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where ff.fid in (11, 12)

-- oprava filtrovani tak aby nebyly zapocteny i duplicity podle id clanku 
select count(distinct a.aid)
from z_field_ford ff
join z_year_field_journal yfj on yfj.fid = ff.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where ff.fid in (11, 12)


-- vypsat pocet clanku ktere jsou v oboru

select ff.name, count(*),count(distinct a.aid)
from z_field_ford ff
join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
where fs.name = 'Natural sciences'
group by ff.fid, ff.name

-- zajimaji nas obory ktere maji vice ja 10k clanku

select ff.name, count(*),count(distinct a.aid)
from z_field_ford ff
join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
group by ff.fid, ff.name
having count(*) > 10000



-- vypiste pocet clanku v prvnim decilu pro kazdou instituci z prahy
-- spatne problem je s filtrovanim ranginku tak aby byl decil
select i.iid ,i.name ,count(*) ,count(distinct a.aid) 
from z_institution i
left join z_article_institution ai on i.iid = ai.iid
left join z_article a on ai.aid = a.aid
left join z_year_field_journal yfj on a.jid = yfj.jid and a.year = yfj.year and yfj.ranking = 'Decil'
where i.town like 'Praha 6%'
group by i.iid, i.name



-- reseni s podotazem

select i.iid ,i.name,
(
	select count(distinct a.aid)
	from z_article_institution ai
	join z_article a on a.aid = ai.aid
	join z_year_field_journal yfj on a.jid = yfj.jid and a.year = yfj.year
	where ai.iid = i.iid and yfj.ranking = 'Decil'
)
from z_institution i
where i.town like 'Praha 6%'



-- zadani jeden ze dvou nize pak jeden tezzsi komibnace agragaci, vice group by
	-- 1, grates per group vrattemi osobu, ktera udelala neco nejvice
	-- 2, dotazy na prunik, najdi osobu, ktera publikovala s radime B. a michalem k., prunik dvou osob
	-- 3, rozdil, naleznete osoby 
	-- 4, statistiky, pro instituce, obory journaly spocinani statistik
	-- instituce ktera ma vice nez 40 publikaci v prvnim decilu a vice nez z q1