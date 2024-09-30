/******************************************************************************
 * 1 Základy SQL, p?íkaz SELECT                                               *
 ******************************************************************************/

/* 1. Vypiste e-mailové e adresy všech neaktivních zakazníku */
select email from customer where active = 0;

/*2. Vypiste názvy a popisy všech film? s klasifikací (atribut rating) G. Vystup bude set?ízen sestupne podle názvu filmu. */
select title, description from film where rating = 'G' order by title desc;

/*3. Vypiste všechny údaje o platbách, které prob?hly v roce 2006 nebo pozd?ji a ?ástka byla menší než 2. */
select * from payment where Year(payment_date) = 2006 and amount > 2;

/*4. Vypiste popisy všech film? klasifikovaných jako G nebo PG */
select description from film where rating in ('G', 'PG');

/* 5. Vypište popisy všech film? klasifikovaných jako G, PG nebo PG-13. */
select * from film where rating in ('G', 'PG', 'PG-13');

/* 6. Vypište popisy všech film?, které nejsou klasifikovány jako G, PG nebo PG-13. */
select description from film where rating not in ('G', 'PG', 'PG-13');

/* 7. Vypište všechny údaje film?, jejichž délka p?esahuje 50 minut a doba výp?j?ky je 3 nebo 5 dní. */
select * from film where length > 50 and rental_duration in  (3, 5);

/*
8. Vypište názvy film?, které obsahují ”RAINBOW“ nebo za?ínají na ”TEXAS“ a jejich délka presahuje 70 minut.
Zamyslete se nad nejednozna?ností formulace této úlohy v p?irozeném jazyce.
*/
select title from film where title like 'TEXAS%' or title like '%RAINBOW%' and length > 70;

/*9. Vypište názvy všech film?, v jejichž popisu se vyskytuje ”And“, jejich délka spadá do intervalu 80 az 90 minut a standardní doba výp?j?ky (atribut rental duration) je liché ?íslo. */
select * from film where description like '%And%' and length between 80 and 90 and rental_duration % 2 = 1;

/*10. Vypište vlastnosti (atribut special features) všech film?, kde ?ástka za náhradu škody (atribut replacement cost) je v intervalu 14 az 16. Zaji?t?te, aby se vlastnosti ve výspisu
neopakovaly. Set?i?te vybrané vlastnosti abecedn?. Zamyslete se, pro? je výsledek i bez explicitního pozadavku na set?ízení jiz abecedn? set?ízeny.*/
select DISTINCT special_features from film where replacement_cost between 14 and 16 order by special_features desc;

/* 11. Vypište všechny údaje film?, jejichž standardní doba výp?j?ky je menší než 4 dny, nebo jsou klasifikovány jako PG. Nesmí však spl?ovat ob? podmínky zárove?. */
select * from film where (rental_duration < 4 or rating = 'PG') and not (rental_duration < 4 and rating = 'PG');

/* 12. Vypište všechny údaje o adresách, které mají vypln?no PS?. */
select * from address where postal_code is not null;

/* 13. Vypište ID všech zákazník?, kte?í aktuáln? mají vyp?j?ený n?jaký film. Dokázali byste spo?ítat, kolik takových zákazník? je? */
select distinct customer_id from rental where return_date is null;
select count(distinct customer_id) as number_of_customers from rental where return_date is null;

/* 14. Pro každé ID platby vypište v samostatných sloupcích rok, m?síc a den, kdy platba probšhla. Sloupce vhodn? pojmenujte. */
select YEAR(payment_date) as YEAR, MONTH(payment_date) as month, DAY(payment_date) as day, payment_id  from payment;

/* 15. Vypište filmy, jejichž délka názvu není 20 znak?. */
select * from film where len(title) != 20;

/* 16. Pro každou ukon?enou výp?j?ku (její ID) vypište dobu jejího trvání v minutách. */
select rental_id, DATEDIFF(MINUTE, rental_date, return_date) as rental_duration_in_min  from rental; 

/* 17. Pro každého aktivního zákazníka vypište jeho celé jméno v jednom sloupci. Výstup tedy bude obsahovat dva sloupce – customer_id a full_name. */
select customer_id, CONCAT(first_name, ' ' ,last_name) as full_name from customer;

/* 18. Pro každou adresu (atribut address) vypište PS?. Jestliže PS? nebude vypln?no, bude se místo n?j zobrazovat text ”(prázdné)“. */
select address, ISNULL(postal_code, '(prázdné)') as PSC from address;

/* 19. Pro všechny uzav?ené výp?j?ky vypište v jednom sloupci interval od – do (tj. ob? data odd?lená poml?kou), kdy výp?j?ka probíhala. */
select rental_id, CONCAT(rental_date, ' - ', return_date) from rental;

/* 20. Pro všechny výp?j?ky vypište v jednom sloupci interval od – do (tj. ob? data odd?lená poml?kou), kdy výp?j?ka probíhala. Pokud výp?j?ka dosud nebyla vrácena, vypište pouze datum od. */
select rental_id, concat(rental_date, isnull(concat(' - ', return_date), '')) as rental_from_to from rental;

/* 21. Vypište po?et všech film? v databázi. */
select count(*) as pocet from film;

/* 22. Vypište po?et r?zných klasifikací film? (atribut rating). */
select count( distinct rating) as cnt from film;

/* 23. Vypište jedním dotazem po?et adres, po?et adres s vypln?ným PS? a po?et r?znyý PS?. */
select COUNT(address) as pocet_address, COUNT(postal_code) as pocet_PSC, COUNT(distinct postal_code) as pocet_PSC_ruzny from address;

/* 24. Vypište nejmenší, nejv?tší a pr?m?rnou délku všech film?. Ov??te si zjišt?nou pr?um?rnou délku pomocí podílu sou?tu a po?tu. */
select min(length) as min_delka, max(length) as max_delka, sum(length) / COUNT(*) as prum_delka  from film;select min(length) as min_length, max(length) as max_length, avg(length) as average_length from film;
select sum(length) as sum_of_length_of_all_movies, count(*) as number_of_movies, cast(sum(length) as float) / count(*) as average_length from film;

/* 25. Vypište po?et a sou?et všech plateb, které byly provedeny v roce 2005. */
select COUNT(*) as pocet, SUM(amount) as celkem from payment where YEAR(payment_date) = 2005;

/* 26. Vypište celkový po?et znak? v názvech všech film?. */
select SUM(LEN(title)) as pocet_znaku from film;
