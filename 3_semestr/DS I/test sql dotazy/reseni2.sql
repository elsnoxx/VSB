select j.jid, j.name, j.issn
from z_journal j
join z_year_field_journal yfj on j.jid = yfj.jid and yfj.year = 2017
join z_field_ford ff on yfj.fid = ff.fid
where ff.name = '3.3 Health Sciences' and exists(
	select *
	from z_journal
	join z_year_field_journal yfj on j.jid = yfj.jid and yfj.year = 2017
	join z_field_ford ff on yfj.fid = ff.fid
	where ff.name = '5.5 Law'
)
order by j.name asc