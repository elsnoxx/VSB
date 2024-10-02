/******************************************************************************
 * 2 Spojování tabulek                                                        *
 ******************************************************************************/
/* 1. Vypište všechny informace o m?stech v?etn? odpovídajících informací o státech, kde se tato m?sta nachází. */
select * from city join country on city.country_id = country.country_id

/* 2. Vypište názvy všech film? v?etn? jejich jazyka. */
select film.title as movie, language.name as language from film join language on film.language_id = language.language_id;


/* 3. Vypište ID všech výp?j?ek zákazníka s p?íjmením SIMPSON. */
select rental.rental_id from rental join customer on customer.customer_id = rental.rental_id where customer.last_name = 'SIMPSON'

/* 4. Vypište adresu (atribut address v tabulce address) zákazníka s p?íjmením SIMPSON. Porovnejte tento p?íklad s p?edchozím co do po?tu ?ádk? ve výsledku. */
select address from customer join address on customer.address_id = address.address_id where customer.last_name = 'SIMPSON'

/* 5. Pro každého zákazníka (jeho jméno a p?íjmení) vypište adresu bydlišt? v?etn? názvu m?sta. */
select customer.first_name, customer.last_name, address.address, city.city 
from customer 
join address on customer.address_id = address.address_id 
join city on address.address_id = city.city_id

/* 6. Pro každého zákazníka (jeho jméno a p?íjmení) vypište název m?sta, kde bydlí. */
select customer.first_name, customer.last_name, city.city 
from customer 
join address on customer.address_id = address.address_id 
join city on address.address_id = city.city_id

/* 7. Vypište ID všech výp?j?ek v?etn? jména zam?stnance, jména zákazníka a názvu filmu. */
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

/* 8. Pro každý film (jeho název) vypište jména a p?íjmení všech herc?, kte?í ve filmu hrají. Kolik ?ádk? bude ve výsledku tohoto dotazu? */
select film.title, actor.first_name, actor.last_name

from film
join film_actor on film.film_id = film_actor.film_id
join actor on film_actor.actor_id = actor.actor_id


/* 9. Pro každého herce (jeho jméno a p?íjmení) vypište jména všech film?, kde herec hrál. ?ím se bude tento dotaz lišit od p?edchozího? Co m?žeme ?íct o operaci vnit?ního spojení tabulek? */
select actor.first_name, actor.last_name, film.title
from actor
join film_actor on film_actor.actor_id = actor.actor_id
join film on film.film_id = film_actor.film_id

/* 10. Vypište názvy všech film? v kategorii ”Horror“. */
select film.title 
from film
join film_category on film_category.film_id = film.film_id
join category on category.category_id = film_category.category_id
where category.name = 'Horror'

/* 11. Pro každý sklad (jeho ID) vypište jméno a p?íjmení jeho správce. Dále vypište adresu skladu a adresu správce (u obou adres sta?í atribut address v tabulce address).
   ?ešení dále rozši?te o výpis adresy v?etn? názvu m?sta a státu. */
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

/* 12. Pro každý film (ID a název) vypište ID všech herc? a ID všech kategorií, do kterých film spadá. Tzn. napište dotaz, jehož výsledkem bude tabulka s atributy film_id, actor_id a category_id, set?ízeno dle film_id.
   Z výsledku pohledem zjist?te, kolik herc? hraje ve filmu s film id = 1, kolik tomuto filmu odpovídá kategorií a kolik je pro tento film celkem ?ádk? ve výsledku. */
   select 
   film.film_id, film.title, actor.actor_id, category.category_id
   from film
   join film_actor on film_actor.film_id = film.film_id
   join actor on actor.actor_id = film_actor.actor_id
   join film_category on film_category.film_id = film.film_id
   join category on category.category_id = film_category.category_id
   order by film.film_id

-- po?et herc? pro film id = 1
select count(distinct actor_id) as actor_count
from film
join film_actor on film.film_id = film_actor.film_id
where film.film_id = 1;

-- po?et kategorií pro film id = 1
select count(distinct category_id) as category_count
from film
join film_category on film.film_id = film_category.film_id
where film.film_id = 1;

-- celkový po?et ?ádk? pro film id = 1
select count(*) as total_rows
from film
join film_actor on film.film_id = film_actor.film_id
join film_category ON film.film_id = film_category.film_id
where film.film_id = 1;



/* 13. Vypište všechny kombinace atribut? ID herce a ID kategorie, kde daný herec hrál ve filmu v dané kategorii.
   Výsledek set?i?te dle ID herce. Dotaz dále rozši?te o výpis jména a p?íjmení herce a názvu kategorie. */
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




/* 14. Vypište jména film?, které p?j?ovna vlastní alespo? v jedné kopii. */
select distinct film.title
from film
join inventory on film.film_id = inventory.inventory_id

/* 15. Zjist?te jména herc?, kte?í hrají v n?jaké komedii (kategorie ”Comedy“). */
select distinct actor.first_name, actor.last_name
from actor
join film_actor on actor.actor_id = film_actor.actor_id
join film_category on film_actor.film_id = film_category.film_id
join category on film_category.category_id = category.category_id
where category.name = 'Comedy';


/* 16. Vypište jména všech zákazník?, kte?í pochází z Itálie a n?kdy m?li nebo mají p?j?ený film s názvem MOTIONS DETAILS. */


/* 17. Zjist?te jména a p?íjmení všech zákazník?, kte?í mají aktuáln? vyp?j?ený n?jaký film, kde hraje herec SEAN GUINESS. */


/* 18. Vypište ID a ?ástku všech plateb a u každé platby uve?te datum výp?j?ky, tj. hodnotu atributu rental_date v tabulce rental. U plateb, které se nevztahují k žádné výp?j?ce bude datum výp?j?ky NULL. */


/* 19. Pro každý jazyk vypište názvy všech film? v daném jazyce. Zajist?te, aby byl jazyk ve výsledku obsažen, i když k n?mu nebude existovat odpovídající film. */


/* 20. Pro každý film (ID a název) vypište jeho jazyk a jeho p?vodní jazyk. */

/* 21. Vypište názvy film?, které si n?kdy p?j?il zákazník TIM CARY, nebo je jejich délka 48 minut. */


/* 22. Vypište názvy film?, které p?j?ovna nevlastní ani v jedné kopii (tj. nejsou obsaženy v inventá?i). */


/* 23. Vypište jména a p?íjmení všech zákazník?, kte?í mají n?jakou nezaplacenou výp?j?ku. */


/* 24. U každého názvu filmu vypište jazyk filmu, pokud jazyk za?íná písmenem ”I“, v opa?ném p?ípad? bude jazyk NULL. */


/* 25. Pro každého zákazníka vypište ID všech plateb s ?ástkou v?tší než 9. U zákazník?, kte?í takovéto platby nemají, bude payment_id rovno NULL. */


/* 26. Pro každou výp?j?ku (její ID) vypište název filmu, pokud obsahuje písmeno ”U“, a m?sto a stát zákazníka, jehož adresa obsahuje písmeno ”A“.
   Podobn? jako v p?edchozích úlohách – jestliže údaj nespl?uje danou podmínku, bude v p?íslušném poli uvedeno NULL. */



/* 27. Vypište všechny dvojice název filmu a p?íjmení zákazníka, kde si zákazník vyp?j?il daný film. Pokud výp?j?ka prob?hla po datu 1.1.2006, bude p?íjmení zákazníka nevypln?né (tj. NULL).
       Z výsledku odstra?te duplicitní ?ádky a set?i?te jej podle názvu filmu. */


