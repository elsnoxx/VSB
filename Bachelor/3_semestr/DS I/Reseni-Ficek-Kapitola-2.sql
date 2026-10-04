/******************************************************************************
 * 2 Spojování tabulek                                                        *
 ******************************************************************************/

/* 1. Vypište všechny informace o městech včetně odpovídajících informací o státech, kde se tato města nachází. */
select * from city 
join country on city.country_id = country.country_id;

/* 2. Vypište názvy všech filmů včetně jejich jazyka. */
select film.title as movie, language.name as language 
from film 
join language on film.language_id = language.language_id;

/* 3. Vypište ID všech výpůjček zákazníka s příjmením SIMPSON. */
select rental.rental_id 
from rental 
join customer on customer.customer_id = rental.customer_id -- opraveno rental.rental_id na customer.customer_id
where customer.last_name = 'SIMPSON';

/* 4. Vypište adresu (atribut address v tabulce address) zákazníka s příjmením SIMPSON. Porovnejte tento příklad s předchozím co do počtu řádků ve výsledku. */
select address.address 
from customer 
join address on customer.address_id = address.address_id 
where customer.last_name = 'SIMPSON';

/* 5. Pro každého zákazníka (jeho jméno a příjmení) vypište adresu bydliště včetně názvu města. */
select customer.first_name, customer.last_name, address.address, city.city 
from customer 
join address on customer.address_id = address.address_id 
join city on address.city_id = city.city_id; -- opraveno address_id na city_id

/* 6. Pro každého zákazníka (jeho jméno a příjmení) vypište název města, kde bydlí. */
select customer.first_name, customer.last_name, city.city 
from customer 
join address on customer.address_id = address.address_id 
join city on address.city_id = city.city_id; -- opraveno address_id na city_id

/* 7. Vypište ID všech výpůjček včetně jména zaměstnance, jména zákazníka a názvu filmu. */
select
    rental.rental_id,
    staff.first_name as staff_first,
    staff.last_name as staff_last,
    customer.first_name as customer_first,
    customer.last_name as customer_last,
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

/* 9. Pro každého herce (jeho jméno a příjmení) vypište jména všech filmů, kde herec hrál. Čím se bude tento dotaz lišit od předchozího? Co můžeme říct o operaci vnitřního spojení tabulek? */
select actor.first_name, actor.last_name, film.title 
from actor 
join film_actor on film_actor.actor_id = actor.actor_id 
join film on film.film_id = film_actor.film_id;

/* 10. Vypište názvy všech filmů v kategorii "Horror". */
select film.title 
from film 
join film_category on film_category.film_id = film.film_id 
join category on category.category_id = film_category.category_id 
where category.name = 'Horror';

/* 11. Pro každý sklad (jeho ID) vypište jméno a příjmení jeho správce. Dále vypište adresu skladu a adresu správce (u obou adres stačí atribut address v tabulce address). Řešení dále rozšiřte o výpis adresy včetně názvu města a státu. */
select
    store.store_id, 
    staff.first_name, staff.last_name,
    store_address.address as store_address,
    manager_address.address as manager_address
from store
join staff on store.manager_staff_id = staff.staff_id
join address as store_address on store.address_id = store_address.address_id
join address as manager_address on staff.address_id = manager_address.address_id;

-- Rozšířené řešení o výpis města a státu
select
    store.store_id, 
    staff.first_name, staff.last_name,
    store_address.address as store_address,
    store_city.city as store_city,
    store_country.country as store_country,
    manager_address.address as manager_address,
    manager_city.city as manager_city,
    manager_country.country as manager_country
from store
join staff on store.manager_staff_id = staff.staff_id
join address as store_address on store.address_id = store_address.address_id
join city as store_city on store_address.city_id = store_city.city_id
join country as store_country on store_city.country_id = store_country.country_id
join address as manager_address on staff.address_id = manager_address.address_id
join city as manager_city on manager_address.city_id = manager_city.city_id
join country as manager_country on manager_city.country_id = manager_country.country_id;

/* 12. Pro každý film (ID a název) vypište ID všech herců a ID všech kategorií, do kterých film spadá. Seřaďte dle film_id. */
select 
    film.film_id, film.title, actor.actor_id, category.category_id 
from film
join film_actor on film_actor.film_id = film.film_id
join actor on actor.actor_id = film_actor.actor_id
join film_category on film_category.film_id = film.film_id
join category on category.category_id = film_category.category_id
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
join film_category on film.film_id = film_category.film_id 
where film.film_id = 1;

/* 13. Vypište všechny kombinace atributů ID herce a ID kategorie, kde daný herec hrál ve filmu v dané kategorii. Seřaďte dle ID herce. */
select distinct 
    actor.actor_id, category.category_id 
from actor 
join film_actor on actor.actor_id = film_actor.actor_id 
join film_category on film_actor.film_id = film_category.film_id 
join category on film_category.category_id = category.category_id 
order by actor.actor_id;

-- Rozšířeno o jméno herce a název kategorie
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
join inventory on film.film_id = inventory.film_id; -- opraveno inventory_id na film_id

/* 15. Zjistěte jména herců, kteří hrají v nějaké komedii (kategorie "Comedy"). */
select distinct actor.first_name, actor.last_name 
from actor 
join film_actor on actor.actor_id = film_actor.actor_id 
join film_category on film_actor.film_id = film_category.film_id 
join category on film_category.category_id = category.category_id 
where category.name = 'Comedy';

/* 16. Vypište jména všech zákazníků, kteří pocházejí z Itálie a někdy měli nebo mají půjčený film s názvem "MOTIONS DETAILS". */
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
where actor.first_name = 'SEAN' and actor.last_name = 'GUINESS' 
and rental.return_date is null;

/* 18. Vypište ID a částku všech plateb a u každé platby uveďte datum výpůjčky, tj. hodnotu atributu rental_date v tabulce rental. U plateb, které se nevztahují 
k žádné výpůjčce bude datum výpůjčky NULL. */
select payment.payment_id, payment.amount, rental.rental_date
from payment
left join rental on payment.rental_id = rental.rental_id;


/* 19. Pro každý jazyk vypište názvy všech filmů v daném jazyce. Zajistěte, aby byl jazyk ve výsledku obsažen, i když k němu nebude existovat odpovídající film. */
select language.name, film.title
from language
left join film on film.language_id = language.language_id

/* 20. Pro každý film (ID a název) vypište jeho jazyk a jeho původní jazyk. */
select film.film_id, film.title, language.name as language, language2.name as language2
from film
join language on language.language_id = film.language_id
left join language as language2 on language2.language_id = film.original_language_id

/* 21. Vypište názvy filmů, které si někdy půjčil zákazník TIM CARY, nebo je jejich délka 48 minut. */
select distinct film.title
from film
join inventory on film.film_id = inventory.film_id
join rental on rental.inventory_id = film.film_id
join customer on customer.customer_id = rental.customer_id
where (customer.first_name = 'TIM' and customer.last_name = 'CARY') or film.length = 48

/* 22. Vypište názvy filmů, které půjčovna nevlastní ani v jedné kopii (tj. nejsou obsaženy v inventáři). */
select film.title
from film
where film.film_id not in (select inventory.film_id from inventory);

