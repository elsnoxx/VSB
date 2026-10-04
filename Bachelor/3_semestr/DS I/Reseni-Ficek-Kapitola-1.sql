/****************************************************************************** 
 * 1 Základy SQL, příkaz SELECT                                                *
 ******************************************************************************/

/* 1. Vypište e-mailové adresy všech neaktivních zákazníků */
select email from customer where active = 0;

/* 2. Vypište názvy a popisy všech filmů s klasifikací (atribut rating) G. Výstup bude seřazen sestupně podle názvu filmu. */
select title, description from film where rating = 'G' order by title desc;

/* 3. Vypište všechny údaje o platbách, které proběhly v roce 2006 nebo později a částka byla menší než 2. */
select * from payment where Year(payment_date) = 2006 and amount > 2;

/* 4. Vypište popisy všech filmů klasifikovaných jako G nebo PG */
select description from film where rating in ('G', 'PG');

/* 5. Vypište popisy všech filmů klasifikovaných jako G, PG nebo PG-13. */
select * from film where rating in ('G', 'PG', 'PG-13');

/* 6. Vypište popisy všech filmů, které nejsou klasifikovány jako G, PG nebo PG-13. */
select description from film where rating not in ('G', 'PG', 'PG-13');

/* 7. Vypište všechny údaje filmů, jejichž délka přesahuje 50 minut a doba výpůjčky je 3 nebo 5 dnů. */
select * from film where length > 50 and rental_duration in (3, 5);

/*
8. Vypište názvy filmů, které obsahují 'RAINBOW' nebo začínají na 'TEXAS' a jejich délka přesahuje 70 minut.
Zamyslete se nad nejednoznačností formulace této úlohy v přirozeném jazyce.
*/
select title from film where title like 'TEXAS%' or title like '%RAINBOW%' and length > 70;

/* 9. Vypište názvy všech filmů, v jejichž popisu se vyskytuje 'And', jejich délka spadá do intervalu 80 až 90 minut a standardní doba výpůjčky (atribut rental duration) je liché číslo. */
select * from film where description like '%And%' and length between 80 and 90 and rental_duration % 2 = 1;

/* 10. Vypište vlastnosti (atribut special features) všech filmů, kde částka za náhradu škody (atribut replacement cost) je v intervalu 14 až 16. Zajistěte, aby se vlastnosti ve výpisu
neopakovaly. Seřaďte vybrané vlastnosti abecedně. Zamyslete se, proč je výsledek i bez explicitního požadavku na seřazení již abecedně seřazen. */
select DISTINCT special_features from film where replacement_cost between 14 and 16 order by special_features desc;

/* 11. Vypište všechny údaje filmů, jejichž standardní doba výpůjčky je menší než 4 dny, nebo jsou klasifikovány jako PG. Nesmí však splňovat obě podmínky zároveň. */
select * from film where (rental_duration < 4 or rating = 'PG') and not (rental_duration < 4 and rating = 'PG');

/* 12. Vypište všechny údaje o adresách, které mají vyplněno PSČ. */
select * from address where postal_code is not null;

/* 13. Vypište ID všech zákazníků, kteří aktuálně mají vypůjčený nějaký film. Dokázali byste spočítat, kolik takových zákazníků je? */
select distinct customer_id from rental where return_date is null;
select count(distinct customer_id) as number_of_customers from rental where return_date is null;

/* 14. Pro každé ID platby vypište v samostatných sloupcích rok, měsíc a den, kdy platba proběhla. Sloupce vhodně pojmenujte. */
select YEAR(payment_date) as YEAR, MONTH(payment_date) as month, DAY(payment_date) as day, payment_id from payment;

/* 15. Vypište filmy, jejichž délka názvu není 20 znaků. */
select * from film where len(title) != 20;

/* 16. Pro každou ukončenou výpůjčku (její ID) vypište dobu jejího trvání v minutách. */
select rental_id, DATEDIFF(MINUTE, rental_date, return_date) as rental_duration_in_min from rental; 

/* 17. Pro každého aktivního zákazníka vypište jeho celé jméno v jednom sloupci. Výstup tedy bude obsahovat dva sloupce – customer_id a full_name. */
select customer_id, CONCAT(first_name, ' ' ,last_name) as full_name from customer;

/* 18. Pro každou adresu (atribut address) vypište PSČ. Jestliže PSČ nebude vyplněno, bude se místo něj zobrazovat text '(prázdný)'. */
select address, ISNULL(postal_code, '(prázdný)') as PSC from address;

/* 19. Pro všechny uzavřené výpůjčky vypište v jednom sloupci interval od do (tj. obě data oddělená pomlčkou), kdy výpůjčka probíhala. */
select rental_id, CONCAT(rental_date, ' - ', return_date) from rental;

/* 20. Pro všechny výpůjčky vypište v jednom sloupci interval od do (tj. obě data oddělená pomlčkou), kdy výpůjčka probíhala. Pokud výpůjčka dosud nebyla vrácena, vypište pouze datum od. */
select rental_id, concat(rental_date, isnull(concat(' - ', return_date), '')) as rental_from_to from rental;

/* 21. Vypište počet všech filmů v databázi. */
select count(*) as pocet from film;

/* 22. Vypište počet různých klasifikací filmů (atribut rating). */
select count(distinct rating) as cnt from film;

/* 23. Vypište jedním dotazem počet adres, počet adres s vyplněným PSČ a počet různých PSČ. */
select COUNT(address) as pocet_address, COUNT(postal_code) as pocet_PSC, COUNT(distinct postal_code) as pocet_PSC_ruzny from address;

/* 24. Vypište nejmenší, největší a průměrnou délku všech filmů. Ověřte si zjištěnou průměrnou délku pomocí podílu součtu a počtu. */
select min(length) as min_delka, max(length) as max_delka, sum(length) / COUNT(*) as prum_delka from film;
select min(length) as min_length, max(length) as max_length, avg(length) as average_length from film;
select sum(length) as sum_of_length_of_all_movies, count(*) as number_of_movies, cast(sum(length) as float) / count(*) as average_length from film;

/* 25. Vypište počet a součet všech plateb, které byly provedeny v roce 2005. */
select COUNT(*) as pocet, SUM(amount) as celkem from payment where YEAR(payment_date) = 2005;

/* 26. Vypište celkový počet znaků v názvech všech filmů. */
select SUM(LEN(title)) as pocet_znaku from film;
