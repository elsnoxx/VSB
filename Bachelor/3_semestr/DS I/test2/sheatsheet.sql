
-- vlozeni dat do tabulky actor
        INSERT INTO actor (first_name, last_name)
        VALUES (’Arnold’, ’Schwarzenegger’);


 -- ziskani hodnot ulozeni do promennych a pote pradni do insertu

        -- Deklarace proměnných
        DECLARE @film_id INT;
        DECLARE @actor_id INT;

        -- Přiřazení hodnot proměnným
        SELECT @film_id = film_id FROM film WHERE title = 'Terminator';
        SELECT @actor_id = actor_id FROM actor WHERE last_name = 'Schwarzenegger';


        -- Použití proměnných ve vložovacím příkazu
        INSERT INTO film_actor (film_id, actor_id) VALUES (@film_id, @actor_id);

-- insert ale data jsou nalezeny pomoci sql selectu

        INSERT INTO film_category (film_id, category_id) VALUES
        (
            (SELECT film_id FROM film WHERE title = ’Termin´ ator’),
            (SELECT category_id FROM category WHERE name = ’Comedy’)
        );



-- ukazka update 

        -- Aktualizace hodnoty ve sloupci
        UPDATE tabulka_jmeno SET sloupec_jmeno = nova_hodnota WHERE podminka;

        -- Příklad
        UPDATE studenti SET vek = 21 WHERE id = 1;

        -- priklad se selectem
        UPDATE film
        SET rental_rate = 2.99, last_update = CURRENT_TIMESTAMP
        WHERE film_id = (SELECT film_id FROM film WHERE title = ’Termin´ ator’);

        -- priklad slozitejsi s IN

        UPDATE film
        SET rental_rate = rental_rate * 1.1
        WHERE film_id IN (
            SELECT film_id
            FROM
            film_actor
            JOIN actor ON film_actor.actor_id = actor.actor_id
            WHERE first_name = ’ZERO’ AND last_name = ’CAGE’
        );


-- IN
-- Operátor IN zkontroluje, zda je hodnota ve specifikovaném seznamu nebo v sadě hodnot vrácených z poddotazu.
-- Často se používá v příkazech WHERE pro filtrování na základě více možných hodnot.
        SELECT title
        FROM film
        WHERE director_id IN (SELECT director_id FROM director WHERE last_name IN ('Spielberg', 'Scorsese'));

-- ALL
-- Operátor ALL je používán k porovnání hodnoty se všemi hodnotami v poddotazu. Používá se většinou s operátory jako >, <, >=, <=, apod.
-- Podmínka s ALL je pravdivá pouze tehdy, když je splněna pro všechny hodnoty vrácené poddotazem.


        SELECT title
        FROM film
        WHERE rating > ALL (SELECT rating FROM film WHERE director_id = 1);

-- Exists
-- Operátor EXISTS zkontroluje, zda poddotaz vrátí alespoň jeden záznam. Vrací hodnotu TRUE, pokud poddotaz vrací alespoň jeden řádek,
-- jinak vrací FALSE. Často se používá ke kontrole existence záznamu na základě podmínky.

        SELECT title
        FROM film f
        WHERE EXISTS (
            SELECT 1
            FROM film_actor fa
            JOIN actor a ON fa.actor_id = a.actor_id
            WHERE fa.film_id = f.film_id AND a.last_name = 'Schwarzenegger'
        );
