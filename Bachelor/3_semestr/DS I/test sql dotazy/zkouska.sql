-- vypsat statistiky o institucích a jejich článcích 
-- (např. počet článků pro každou instituci, průměrné hodnocení časopisů atd.)


SELECT i.iid, i.name AS institution_name, COUNT(a.aid) AS article_count
FROM z_institution i
JOIN z_article_institution ai ON ai.iid = i.iid
JOIN z_article a ON a.aid = ai.aid
GROUP BY i.iid, i.name
ORDER BY article_count DESC;
