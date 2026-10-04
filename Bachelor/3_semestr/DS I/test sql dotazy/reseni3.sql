WITH t as (
select  ff.fid ,
		ff.name, 
		count(a.aid) as pocet

from z_field_ford ff

join z_field_of_science fs on ff.sid = fs.sid
join z_year_field_journal yfj on ff.fid = yfj.fid
join z_article a on a.jid = yfj.jid and a.year = yfj.year
group by ff.fid ,ff.name

)

select t.fid, t.name from t
where pocet = (
	select max(pocet)
	from t
)