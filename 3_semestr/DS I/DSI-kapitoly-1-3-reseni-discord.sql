/******************************************************************************
 * 1 Základy SQL, příkaz SELECT                                               *
 ******************************************************************************/
/* 1. Vypište e-mailové adresy všech neaktivních zákazníků. */
select email from customer where active = 0;

/* 2. Vypište názvy a popisy všech filmů s klasifikací (atribut rating) G. Výstup bude setřízen sestupně podle názvu filmu. */
select title, description from film where rating = 'G' order by title desc;

/* 3. Vypište všechny údaje o platbách, které proběhly v roce 2006 nebo později a částka byla menší než 2. */
select * from payment where payment_date >= '2006-01-01' and amount < 2;

/* 4. Vypište popisy všech filmů klasifikovaných jako G nebo PG. */
select description from film where rating in ('G', 'PG');

/* 5. Vypište popisy všech filmů klasifikovaných jako G, PG nebo PG-13. */
select description from film where rating in ('G', 'PG', 'PG-13');

/* 6. Vypište popisy všech filmů, které nejsou klasifikovány jako G, PG nebo PG-13. */
select description from film where rating not in ('G', 'PG', 'PG-13');

/* 7. Vypište všechny údaje filmů, jejichž délka přesahuje 50 minut a doba výpůjčky je 3 nebo 5 dní. */
select * from film where length > 50 and rental_duration in (3, 5);

/* 8. Vypište názvy filmů, které obsahují ”RAINBOW“ nebo začínají na ”TEXAS“ a jejich délka přesahuje 70 minut. Zamyslete senad nejednoznačností formulace této úlohy v přirozeném jazyce. */
select title from film where (title LIKE '%RAINBOW%' OR title LIKE 'TEXAS%') and length > 70;
-- a proto Bůh vymyslel závorky: Vypište názvy filmů, které (obsahují ”RAINBOW“ nebo začínají na ”TEXAS“) a (jejich délka přesahuje 70 minut)...

/* 9. Vypište názvy všech filmů, v jejichž popisu se vyskytuje ”And“, jejich délka spadá do intervalu 80 až 90 minut a standardní doba výpůjčky (atribut rental_duration) je liché číslo. */
select title from film where description like '%And%' and (length >=80 and length <=90) and rental_duration % 2 = 1;
select title from film where description like '%And%' and length between 80 and 90 and rental_duration % 2 = 1;

/* 10. Vypište vlastnosti (atribut special_features) všech filmů, kde částka za náhradu škody (atribut replacement_cost) je v intervalu 14 až 16.
   Zajistěte, aby se vlastnosti ve výpisu neopakovaly. Setřiďte vybrané vlastnosti abecedně. Zamyslete se, proč je výsledek i bez explicitního požadavku na setřízení již abecedně setřízený. */
   -- protože takhle náhodou jsou data za sebou v DB, bez 'order by' se nelze na jakékoliv setřízení _v žádném případě_ spolehnout
select distinct special_features from film where replacement_cost between 14 and 16 order by special_features;

/* 11. Vypište všechny údaje filmů, jejichž standardní doba výpůjčky je menší než 4 dny, nebo jsou klasifikovány jako PG. Nesmí však splňovat obě podmínky zároveň. */
select * from film where (rental_duration < 4 or rating = 'PG') and not (rental_duration < 4 and rating = 'PG');

/* 12. Vypište všechny údaje o adresách, které mají vyplněno PSČ. */
select * from address where postal_code is not null;

/* 13. Vypište ID všech zákazníků, kteří aktuálně mají vypůjčený nějaký film. Dokázali byste spočítat, kolik takových zákazníků je? */
select distinct customer_id from rental where return_date is null;
select count(distinct customer_id) as number_of_customers from rental where return_date is null;

/* 14. Pro každé ID platby vypište v samostatných sloupcích rok, měsíc a den, kdy platba probšhla. Sloupce vhodně pojmenujte. */
select payment_id, year(payment_date) as 'payment year', month(payment_date) as 'payment month', day(payment_date) as 'payment day' from payment;

/* 15. Vypište filmy, jejichž délka názvu není 20 znaků. */
select * from film where len(title) <> 20;

/* 16. Pro každou ukončenou výpůjčku (její ID) vypište dobu jejího trvání v minutách. */
select rental_id, datediff(minute, rental_date, return_date) as rental_duration_in_min from rental where return_date is not null;

/* 17. Pro každého aktivního zákazníka vypište jeho celé jméno v jednom sloupci. Výstup tedy bude obsahovat dva sloupce – customer_id a full_name. */
select customer_id, concat(first_name, ' ', last_name) as full_name from customer where active = 1;

/* 18. Pro každou adresu (atribut address) vypište PSČ. Jestliže PSČ nebude vyplněno, bude se místo něj zobrazovat text ”(prázdné)“. */
select address, coalesce(postal_code, '(prázdné)') as postal_code from address;

/* 19. Pro všechny uzavřené výpůjčky vypište v jednom sloupci interval od – do (tj. obě data oddělená pomlčkou), kdy výpůjčka probíhala. */
select rental_id, concat(rental_date, ' - ', return_date) as rental_from_to from rental where return_date is not null;

/* 20. Pro všechny výpůjčky vypište v jednom sloupci interval od – do (tj. obě data oddělená pomlčkou), kdy výpůjčka probíhala. Pokud výpůjčka dosud nebyla vrácena, vypište pouze datum od. */
select rental_id, concat(rental_date, isnull(concat(' - ', return_date), '')) as rental_from_to from rental;

select rental_id,
	case
		when return_date is not null
		then concat(try_convert(varchar, rental_date, 120), ' - ', try_convert(varchar, return_date, 120))
		else try_convert(varchar, rental_date, 120)
	end as rental_from_to
from rental;

/* 21. Vypište počet všech filmů v databázi. */
select count(*) as number_of_movies from film;

/* 22. Vypište počet různých klasifikací filmů (atribut rating). */
select count(distinct rating) as number_of_different_movie_classifications from film;

/* 23. Vypište jedním dotazem počet adres, počet adres s vyplněným PSČ a počet různyý PSČ. */
select count(*) as number_of_addresses, count(postal_code) as number_of_addresses_with_postal_code, count(distinct postal_code) as number_of_different_postal_codes from address;

/* 24. Vypište nejmenší, největší a průměrnou délku všech filmů. Ověřte si zjištěnou průuměrnou délku pomocí podílu součtu a počtu. */
select min(length) as min_length, max(length) as max_length, avg(length) as average_length from film;
select sum(length) as sum_of_length_of_all_movies, count(*) as number_of_movies, cast(sum(length) as float) / count(*) as average_length from film;

/* 25. Vypište počet a součet všech plateb, které byly provedeny v roce 2005. */
select count(*) as number_of_payments, sum(amount) as sum_of_payments from payment where year(payment_date) = 2005;

/* 26. Vypište celkový počet znaků v názvech všech filmů. */
select sum(len(title)) as total_number_of_characters_in_titles_of_all_movies from film;



/******************************************************************************
 * 2 Spojování tabulek                                                        *
 ******************************************************************************/
/* 1. Vypište všechny informace o městech včetně odpovídajících informací o státech, kde se tato města nachází. */
select city.*, country.* from city join country on city.country_id = country.country_id;

/* 2. Vypište názvy všech filmů včetně jejich jazyka. */
select film.title as movie, language.name as language from film join language on film.language_id = language.language_id;

/* 3. Vypište ID všech výpůjček zákazníka s příjmením SIMPSON. */
select rental_id from rental join customer on rental.customer_id = customer.customer_id where customer.last_name = 'SIMPSON';

/* 4. Vypište adresu (atribut address v tabulce address) zákazníka s příjmením SIMPSON. Porovnejte tento příklad s předchozím co do počtu řádků ve výsledku. */
select address from address join customer on address.address_id = customer.address_id where customer.last_name = 'SIMPSON';

/* 5. Pro každého zákazníka (jeho jméno a příjmení) vypište adresu bydliště včetně názvu města. */
select customer.first_name, customer.last_name, address.address, city.city
from customer
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id;

/* 6. Pro každého zákazníka (jeho jméno a příjmení) vypište název města, kde bydlí. */
select customer.first_name, customer.last_name, city.city
from customer
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id;

/* 7. Vypište ID všech výpůjček včetně jména zaměstnance, jména zákazníka a názvu filmu. */
select
	rental.rental_id,
	staff.first_name as STAFF_first,
	staff.last_name as STAFF_last,
	customer.first_name as CUSTOMER_first,
	customer.last_name as CUSTOMER_last,
	film.title
from rental
join staff on rental.staff_id = staff.staff_id
join customer on rental.customer_id = customer.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join film on inventory.film_id = film.film_id;

/* 8. Pro každý film (jeho název) vypište jména a příjmení všech herců, kteří ve filmu hrají. Kolik řádků bude ve výsledku tohoto dotazu? */
select film.title, actor.first_name, actor.last_name
from film
join film_actor on film.film_id = film_actor.film_id
join actor on film_actor.actor_id = actor.actor_id;

select count(*) as total_number_of_actors_in_all_movies from film_actor;

/* 9. Pro každého herce (jeho jméno a příjmení) vypište jména všech filmů, kde herec hrál. Čím se bude tento dotaz lišit od předchozího? Co můžeme říct o operaci vnitřního spojení tabulek? */
select actor.first_name, actor.last_name, film.title
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film on film_actor.film_id = film.film_id;
-- v předchozím dotazu byla hlavní entitou film, v tomto dotazu je hlavní entitou actor.
-- inner join je u obou dotazů mezi tabulkami film/film_actor/actor, resp. actor/film_actor/film - jedná se tedy o shodné výstupy, jen výsledek předchozího dotazu je uspořádán podle filmů a výsledek tohoto dotazu je uspořádán podle herců.

/* 10. Vypište názvy všech filmů v kategorii ”Horror“. */
select film.title
from film
join film_category on film.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
where category.name = 'Horror';

/* 11. Pro každý sklad (jeho ID) vypište jméno a příjmení jeho správce. Dále vypište adresu skladu a adresu správce (u obou adres stačí atribut address v tabulce address).
   Řešení dále rozšiřte o výpis adresy včetně názvu města a státu. */
select
	store.store_id, staff.first_name, staff.last_name,
	store_address.address as STORE_address,
	manager_address.address as MANAGER_address
from store
join staff on store.manager_staff_id = staff.staff_id
join address as store_address on store.address_id = store_address.address_id
join address as manager_address on staff.address_id = manager_address.address_id;

select
	store.store_id, staff.first_name, staff.last_name,
	store_address.address as STORE_address,
	store_city.city as STORE_city,
	store_country.country as STORE_country,
	manager_address.address as MANAGER_address,
	manager_city.city as MANAGER_city,
	manager_country.country as MANAGER_country
from store
join staff on store.manager_staff_id = staff.staff_id
join address as store_address on store.address_id = store_address.address_id
join city as store_city on store_address.city_id = store_city.city_id
join country as store_country on store_city.country_id = store_country.country_id
join address as manager_address on staff.address_id = manager_address.address_id
join city as manager_city on manager_address.city_id = manager_city.city_id
join country as manager_country on manager_city.country_id = manager_country.country_id;

/* 12. Pro každý film (ID a název) vypište ID všech herců a ID všech kategorií, do kterých film spadá. Tzn. napište dotaz, jehož výsledkem bude tabulka s atributy film_id, actor_id a category_id, setřízeno dle film_id.
   Z výsledku pohledem zjistěte, kolik herců hraje ve filmu s film id = 1, kolik tomuto filmu odpovídá kategorií a kolik je pro tento film celkem řádků ve výsledku. */
select film.film_id, film.title, actor.actor_id, category.category_id
from film
join film_actor on film.film_id = film_actor.film_id
join actor on film_actor.actor_id = actor.actor_id
join film_category on film.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
order by film.film_id;

-- počet herců pro film id = 1
select count(distinct actor_id) as actor_count
from film
join film_actor on film.film_id = film_actor.film_id
where film.film_id = 1;

-- počet kategorií pro film id = 1
select count(distinct category_id) as category_count
from film
join film_category on film.film_id = film_category.film_id
where film.film_id = 1;

-- celkový počet řádků pro film id = 1
select count(*) as total_rows
from film
join film_actor on film.film_id = film_actor.film_id
join film_category ON film.film_id = film_category.film_id
where film.film_id = 1;

/* 13. Vypište všechny kombinace atributů ID herce a ID kategorie, kde daný herec hrál ve filmu v dané kategorii.
   Výsledek setřiďte dle ID herce. Dotaz dále rozšiřte o výpis jména a příjmení herce a názvu kategorie. */
select distinct
	actor.actor_id,	category.category_id
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film_category on film_actor.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
order by actor.actor_id;

select distinct
	actor.actor_id, actor.first_name, actor.last_name,
	category.category_id,
	category.name as category_name
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film_category on film_actor.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
order by actor.actor_id;

/* 14. Vypište jména filmů, které půjčovna vlastní alespoň v jedné kopii. */
select distinct film.title
from film
join inventory on film.film_id = inventory.film_id;

/* 15. Zjistěte jména herců, kteří hrají v nějaké komedii (kategorie ”Comedy“). */
select distinct actor.first_name, actor.last_name
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film_category on film_actor.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
where category.name = 'Comedy';

/* 16. Vypište jména všech zákazníků, kteří pochází z Itálie a někdy měli nebo mají půjčený film s názvem MOTIONS DETAILS. */
select distinct customer.first_name, customer.last_name
from customer
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
join country on city.country_id = country.country_id
join rental on customer.customer_id = rental.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join film on inventory.film_id = film.film_id
where country.country = 'Italy' and film.title = 'MOTIONS DETAILS';

/* 17. Zjistěte jména a příjmení všech zákazníků, kteří mají aktuálně vypůjčený nějaký film, kde hraje herec SEAN GUINESS. */
select distinct customer.first_name, customer.last_name
from customer
join rental on customer.customer_id = rental.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join film on inventory.film_id = film.film_id
join film_actor on film.film_id = film_actor.film_id
join actor on film_actor.actor_id = actor.actor_id
where
	rental.return_date is null -- aktuálně vypůjčené filmy
	and actor.first_name = 'SEAN'
	and actor.last_name = 'GUINESS';

/* 18. Vypište ID a částku všech plateb a u každé platby uveďte datum výpůjčky, tj. hodnotu atributu rental_date v tabulce rental. U plateb, které se nevztahují k žádné výpůjčce bude datum výpůjčky NULL. */
select payment.payment_id, payment.amount, rental.rental_date
from payment
left join rental on payment.rental_id = rental.rental_id;

/* 19. Pro každý jazyk vypište názvy všech filmů v daném jazyce. Zajistěte, aby byl jazyk ve výsledku obsažen, i když k němu nebude existovat odpovídající film. */
select language.name as language, film.title
from language
left join film on language.language_id = film.language_id;

/* 20. Pro každý film (ID a název) vypište jeho jazyk a jeho původní jazyk. */
select
	film.film_id, film.title,
	language.name as language,
	original_language.name as original_language
from film
join language on film.language_id = language.language_id
left join language as original_language on film.original_language_id = original_language.language_id;

/* 21. Vypište názvy filmů, které si někdy půjčil zákazník TIM CARY, nebo je jejich délka 48 minut. */
select distinct film.title
from film
join inventory on film.film_id = inventory.film_id
join rental on inventory.inventory_id = rental.inventory_id
join customer on rental.customer_id = customer.customer_id
where
	(customer.first_name = 'TIM' and customer.last_name = 'CARY')
	or film.length = 48;

/* 22. Vypište názvy filmů, které půjčovna nevlastní ani v jedné kopii (tj. nejsou obsaženy v inventáři). */
select film.title
from film
where film.film_id not in (select inventory.film_id from inventory);

select film.title
from film
left join inventory on film.film_id = inventory.film_id
where inventory.film_id is null;

/* 23. Vypište jména a příjmení všech zákazníků, kteří mají nějakou nezaplacenou výpůjčku. */
select distinct customer.first_name, customer.last_name
from customer
join rental on customer.customer_id = rental.customer_id
left join payment on rental.rental_id = payment.rental_id
where payment.rental_id is null;

/* 24. U každého názvu filmu vypište jazyk filmu, pokud jazyk začíná písmenem ”I“, v opačném případě bude jazyk NULL. */
select film.title,
	case
		when language.name like 'I%'
		then language.name
		else null
	end as language
from film
left join language on film.language_id = language.language_id;

select film.title,
	(
		select language.name from language
		where language.language_id = film.language_id and language.name like 'I%'
	) as language
from film;

/* 25. Pro každého zákazníka vypište ID všech plateb s částkou větší než 9. U zákazníků, kteří takovéto platby nemají, bude payment_id rovno NULL. */
select customer.first_name, customer.last_name, payment_id
from customer
left join payment on customer.customer_id = payment.customer_id and payment.amount > 9;

/* 26. Pro každou výpůjčku (její ID) vypište název filmu, pokud obsahuje písmeno ”U“, a město a stát zákazníka, jehož adresa obsahuje písmeno ”A“.
   Podobně jako v předchozích úlohách – jestliže údaj nesplňuje danou podmínku, bude v příslušném poli uvedeno NULL. */
select rental.rental_id, film.title, city.city, country.country
from rental
left join inventory on rental.inventory_id = inventory.inventory_id
left join film on inventory.film_id = film.film_id AND film.title like '%U%'
left join customer on rental.customer_id = customer.customer_id
left join address on customer.address_id = address.address_id
left join city on address.city_id = city.city_id and address.address like '%A%'
left join country on city.country_id = country.country_id;

select rental.rental_id,
	case
		when film.title like '%U%'
		then film.title
		else null
	end as film_title,
	case
		when address.address like '%A%'
		then city.city
		else null
	end as city,
	case
		when address.address like '%A%'
		then country.country
		else null
	end as country
from rental
join inventory on rental.inventory_id = inventory.inventory_id
join film on inventory.film_id = film.film_id
join customer on rental.customer_id = customer.customer_id
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
join country on city.country_id = country.country_id;

/* 27. Vypište všechny dvojice název filmu a příjmení zákazníka, kde si zákazník vypůjčil daný film. Pokud výpůjčka proběhla po datu 1.1.2006, bude příjmení zákazníka nevyplněné (tj. NULL).
       Z výsledku odstraňte duplicitní řádky a setřiďte jej podle názvu filmu. */
select distinct film.title, 
	case
		when rental.rental_date > '2006-01-01' THEN null
		else customer.last_name 
	end as last_name
FROM rental
join inventory on rental.inventory_id = inventory.inventory_id
join film on inventory.film_id = film.film_id
join customer on rental.customer_id = customer.customer_id
order by film.title;



/******************************************************************************
 * 3 Agregační funkce a shlukování                                            *
 ******************************************************************************/
 /* 1. Vypište počty filmů pro jednotlivé klasifikace (atribut rating). */
select
	film.rating,
	count(*) as film_count
from film
group by film.rating;

/* 2. Pro každé ID zákazníka vypište počet jeho příjmení. Je ve výsledku něco překvapivého? */ --> NENÍ
select
	customer.customer_id,
	count(customer.last_name) as last_name_count
from customer
group by customer.customer_id;

/* 3. Vypište ID zákazníků setřízená podle součtu všech jejich plateb. Zákazníky, kteří neprovedli žádnou platbu neuvažujte. */
select
	customer.customer_id,
	sum(payment.amount) as total_payments
from customer
join payment on customer.customer_id = payment.customer_id
group by customer.customer_id
order by total_payments;

/* 4. Pro každé jméno a příjmení herce vypište počet herců s takovým jménem a příjmením. Výsledek setřiďte dle počtu sestupně. */
-- pro herce se stejným jménem
select first_name, '' as last_name, count(*) as actor_count
from actor
group by first_name
union
-- pro herce se stejným příjmením
select '' as first_name, last_name, count(*) as actor_count
from actor
group by last_name
order by actor_count desc;

/* 5. Vypište součty všech plateb za jednotlivé roky a měsíce. Výsledek uspořádejte podle roků a měsíců. */
select
	year(payment_date) as year,
    month(payment_date) as month,
    sum(amount) as total_amount
from payment
group by
    year(payment_date),
    month(payment_date)
order by year, month;

/* 6. Vypište ID skladů s více než 2 300 kopiemi filmů. */
select store.store_id
from store
join inventory on store.store_id = inventory.store_id
group by store.store_id
having count(inventory.inventory_id) > 2300;

-- za pomocí subdotazu bez použití "having"
select store_id
from (
	select store.store_id, count(inventory.inventory_id) as total_copies
    from store
    join inventory ON store.store_id = inventory.store_id
    group by store.store_id
) as store_counts
where total_copies > 2300;

/* 7. Vypište ID jazyků, pro které je nejkratší film delší než 46 minut. */
select language_id from film group by language_id having min(length) > 46;

/* 8. Vypište roky a měsíce plateb, kdy byl součet plateb větší než 20 000. */
select
	year(payment_date) as year,
    month(payment_date) as month,
    sum(amount) as total_amount
from payment
group by
    year(payment_date),
    month(payment_date)
having sum(amount) > 20000;

/* 9. Vypište klasifikace filmů (atribut rating), jejichž délka je menší než 50 minut a celková délka v dané klasifikaci je větší než 250 minut. Výsledek setřiďte sestupně podle abecedy. */
select rating from film where length < 50 group by rating having sum(length) > 250 order by rating desc;

/* 10. Vypište pro jednotlivá ID jazyků počet filmů. Vynechejte jazyky, které nemají žádný film. */
select
	language_id,
	count(film_id) as film_count
from film
group by language_id
having count(film_id) > 0;

/* 11. Vypište názvy jazyků a k nim počty filmů. Vynechejte jazyky, které nemají žádný film. */
select
	language.name as language_name,
	count(film.film_id) as film_count
from language
join film on language.language_id = film.language_id
group by language.language_id, language.name;

/* 12. Vypište názvy všech jazyků a k nim počty filmů. Ve výsledku budou zahrnuty i ty jazyky, které nemají žádný film. */
select
	language.name as language_name,
	count(film.film_id) as film_count
from language
left join film on language.language_id = film.language_id
group by language.language_id, language.name;

/* 13. Vypište pro jednotlivé zákazníky (jejich ID, jméno a příjmení) počty jejich výpůjček. */
select
	customer.customer_id,
	customer.first_name,
	customer.last_name,
	count(rental.rental_id) as rental_count
from customer
left join rental on customer.customer_id = rental.customer_id
group by customer.customer_id, customer.first_name, customer.last_name;

/* 14. Vypište pro jednotlivé zákazníky (jejich ID, jméno a příjmení) počty různých filmů, které si vypůjčili. */
select
	customer.customer_id,
	customer.first_name,
	customer.last_name,
	count(distinct inventory.film_id) as unique_movie_count
from customer
left join rental on customer.customer_id = rental.customer_id
left join inventory on rental.inventory_id = inventory.inventory_id
group by customer.customer_id, customer.first_name, customer.last_name;

/* 15. Vypište jména a příjmení herců, kteří hrají ve více než 20-ti filmech. */
select
	actor.first_name,
	actor.last_name
from actor
join film_actor on actor.actor_id = film_actor.actor_id
group by actor.actor_id, actor.first_name, actor.last_name
having count(film_actor.film_id) > 20;

/* 16. Pro každého zákazníka vypište, kolik celkem utratil za výpůjčky filmů a jaké byla jeho nejmenší, největší a průuměrné částka platby. */
select
	customer.first_name,
	customer.last_name,
	sum(payment.amount) as total_spent,
	min(payment.amount) as min_payment,
	max(payment.amount) as max_payment,
	avg(payment.amount) as average_payment
from customer
join payment on customer.customer_id = payment.customer_id
group by customer.customer_id, customer.first_name, customer.last_name;

/* 17. Vypište pro každou kategorii průuměrnou délku filmu. */
select
	category.name as category,
	avg(film.length) as average_movie_length
from category
join film_category on category.category_id = film_category.category_id
join film on film_category.film_id = film.film_id
group by category.category_id, category.name;

/* 18. Pro každý film vypište, jaký byl celkový příjem z výpůjček. Vypište jen filmy, kde byl celkový příjem větší než 100. */
select
	film.title,
	sum(payment.amount) as total_revenue
from film
join inventory on film.film_id = inventory.film_id
join rental on inventory.inventory_id = rental.inventory_id
join payment on rental.rental_id = payment.rental_id
group by film.film_id, film.title
having sum(payment.amount) > 100;

/* 19. Pro každého herce vypište, v kolika různých kategoriích filmů hraje. */
select
	actor.first_name,
	actor.last_name,
	count(distinct film_category.category_id) as category_count
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film_category on film_actor.film_id = film_category.film_id
group by actor.actor_id, actor.first_name, actor.last_name;

/* 20. Vypište adresy zákazníků (atribut address.address) včetně názvu města a státu, kde ve filmech, které si zákazníci půjčili, hrálo dohromady alespoň 40 různých herců. */
select
	address.address,
	city.city,
	country.country
from customer
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
join country on city.country_id = country.country_id
join rental on customer.customer_id = rental.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join film_actor on inventory.film_id = film_actor.film_id
group by customer.customer_id, address.address, city.city, country.country
having count(distinct film_actor.actor_id) >= 40;

/* 21. Pro všechny filmy (ID a název) spadající do kategorie ”Horror“ uveďte, v kolika různých městech bydlí zákazníci, kteří si daný film někdy půjčili. */
-- bez vnořeného selectu, join tabulky navíc (category) -> rychlejší
select
	film.film_id,
    film.title,
    count(distinct city.city_id) as city_count
from film
join film_category on film.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
join inventory on film.film_id = inventory.film_id
join rental on inventory.inventory_id = rental.inventory_id
join customer on rental.customer_id = customer.customer_id
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
where category.name = 'Horror'
group by film.film_id, film.title;

-- vnořený select místo join
select
	film.film_id,
	film.title,
	count(distinct city.city_id) as city_count
from film
join film_category on film.film_id = film_category.film_id
join inventory on film.film_id = inventory.film_id
join rental on inventory.inventory_id = rental.inventory_id
join customer on rental.customer_id = customer.customer_id
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
where film_category.category_id = (select category_id from category where name = 'Horror')
group by film.film_id, film.title;

/* 22. Pro všechny zákazníky z Polska vypište, do kolika různých kategorií spadají filmy, které si tito zákazníci vypůjčili. */
select
    customer.first_name,
    customer.last_name,
    count(distinct film_category.category_id) as category_count
from customer
join address on customer.address_id = address.address_id
join city on address.city_id = city.city_id
join country on city.country_id = country.country_id
join rental on customer.customer_id = rental.customer_id
join inventory on rental.inventory_id = inventory.inventory_id
join film_category on inventory.film_id = film_category.film_id
where country.country = 'Poland'
group by customer.customer_id, customer.first_name, customer.last_name;

/* 23. Vypište názvy všech jazyků, k nim počty filmů v daném jazyce, které jsou delší než 350 minut. */
select
	language.name as language,
	count(film.film_id) as film_count
from language
left join film on language.language_id = film.language_id and film.length > 350
group by language.language_id, language.name;

/* 24. Vypište, kolik jednotliví zákazníci utratili za výpůjčky, které začaly v měsíci červnu. */
select
	customer.first_name,
	customer.last_name,
	sum(payment.amount) as total_spent
from customer
join rental on customer.customer_id = rental.customer_id
join payment on rental.rental_id = payment.rental_id
where month(rental.rental_date) = 6
group by customer.customer_id, customer.first_name, customer.last_name;

/* 25. Vypište seznam kategorií setřízený podle počtu filmů, jejichž jazyk začíná písmenem ”E“. */
select
	category.name as category,
	count(film.film_id) as film_count
from category
join film_category on category.category_id = film_category.category_id
join film on film_category.film_id = film.film_id
join language on film.language_id = language.language_id
where language.name like 'E%'
group by category.category_id, category.name
order by count(film.film_id);

/* 26. Vypište názvy filmů s délkou menší než 50 minut, které si zákazníci s příjmením BELL půjčili přesně 1x. */
select film.title
from film
join inventory on film.film_id = inventory.film_id
join rental on inventory.inventory_id = rental.inventory_id
join customer on rental.customer_id = customer.customer_id
where
	film.length < 50
	and customer.last_name = 'BELL'
group by film.film_id, film.title
having count(rental.rental_id) = 1;
