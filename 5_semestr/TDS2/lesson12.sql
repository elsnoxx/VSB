/*
===============================================================================
LEKCE 12: DYNAMICKÉ SQL A OPTIMALIZACE VÝKONU
===============================================================================
Obsah:
1. DBMS_SQL pro dynamicky generované SQL
2. EXECUTE IMMEDIATE s různými variantami
3. ALTER COMPILE pro procedury, funkce a package
4. Optimalizace výkonu: NOCOPY, DETERMINISTIC, FORALL, BULK COLLECT
===============================================================================
*/


-- =============================================
-- 2. DYNAMICKÉ SQL POMOCÍ DBMS_SQL
-- =============================================

CREATE OR REPLACE PROCEDURE demo_dbms_sql AS
    v_cursor_id     INTEGER;
    v_sql           VARCHAR2(1000);
    v_rows_processed NUMBER;
    v_guest_id      NUMBER;
    v_firstname     VARCHAR2(100);
    v_lastname      VARCHAR2(100);
    v_guest_type    VARCHAR2(50);
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO DBMS_SQL ===');
    
    -- 1. Otevření kurzoru
    v_cursor_id := DBMS_SQL.OPEN_CURSOR;
    
    -- 2. Příprava SQL příkazu
    v_sql := 'SELECT guest_id, firstname, lastname, guest_type FROM Guest WHERE guest_type = :type';
    DBMS_SQL.PARSE(v_cursor_id, v_sql, DBMS_SQL.NATIVE);
    
    -- 3. Binding parametrů
    DBMS_SQL.BIND_VARIABLE(v_cursor_id, ':type', 'VIP');
    
    -- 4. Definice výstupních sloupců
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 1, v_guest_id);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 2, v_firstname, 100);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 3, v_lastname, 100);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 4, v_guest_type, 50);
    
    -- 5. Vykonání
    v_rows_processed := DBMS_SQL.EXECUTE(v_cursor_id);
    
    -- 6. Načítání výsledků
    DBMS_OUTPUT.PUT_LINE('VIP hosté:');
    WHILE DBMS_SQL.FETCH_ROWS(v_cursor_id) > 0 LOOP
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 1, v_guest_id);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 2, v_firstname);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 3, v_lastname);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 4, v_guest_type);
        
        DBMS_OUTPUT.PUT_LINE(v_guest_id || ': ' || v_firstname || ' ' || v_lastname || ' (' || v_guest_type || ')');
    END LOOP;
    
    -- 7. Zavření kurzoru
    DBMS_SQL.CLOSE_CURSOR(v_cursor_id);
    
EXCEPTION
    WHEN OTHERS THEN
        IF DBMS_SQL.IS_OPEN(v_cursor_id) THEN
            DBMS_SQL.CLOSE_CURSOR(v_cursor_id);
        END IF;
        RAISE;
END;
/

-- =============================================
-- 3. EXECUTE IMMEDIATE - RŮZNÉ VARIANTY
-- =============================================

CREATE OR REPLACE PROCEDURE demo_execute_immediate AS
    v_sql           VARCHAR2(1000);
    v_count         NUMBER;
    v_avg_price     NUMBER;
    v_max_price     NUMBER;
    v_guest_type    VARCHAR2(50) := 'VIP';
    v_min_price     NUMBER := 2000;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO EXECUTE IMMEDIATE ===');
    
    -- 1. Jednoduchý SELECT INTO
    EXECUTE IMMEDIATE 'SELECT COUNT(*) FROM Guest' INTO v_count;
    DBMS_OUTPUT.PUT_LINE('Celkový počet hostů: ' || v_count);
    
    -- 2. SELECT s parametry USING IN
    EXECUTE IMMEDIATE 
        'SELECT AVG(accommodation_price), MAX(accommodation_price) FROM Reservation WHERE accommodation_price > :1'
        INTO v_avg_price, v_max_price
        USING IN v_min_price;
    
    DBMS_OUTPUT.PUT_LINE('Rezervace s cenou nad ' || v_min_price || ':');
    DBMS_OUTPUT.PUT_LINE('  Průměrná cena: ' || ROUND(v_avg_price, 2));
    DBMS_OUTPUT.PUT_LINE('  Maximální cena: ' || v_max_price);
    
    -- 3. UPDATE s vstupními parametry
    EXECUTE IMMEDIATE 
        'UPDATE Guest SET guest_type = :new_type WHERE guest_type = :old_type AND guest_id > :min_id'
        USING IN 'Premium', 'Regular', 50;
    
    DBMS_OUTPUT.PUT_LINE('Aktualizováno ' || SQL%ROWCOUNT || ' záznamů hostů');
    
    -- 4. INSERT s RETURNING klauzulí
    DECLARE
        v_new_id NUMBER;
    BEGIN
        EXECUTE IMMEDIATE 
            'INSERT INTO Service (name, description) VALUES (:name, :desc) RETURNING service_id INTO :new_id'
            USING IN 'Room Service', 'In-room dining service'
            RETURNING INTO v_new_id;
            
        DBMS_OUTPUT.PUT_LINE('Vložena nová služba s ID: ' || v_new_id);
    END;
    
    -- 5. DDL příkaz
    BEGIN
        EXECUTE IMMEDIATE 'CREATE TABLE temp_reservation_stats (guest_type VARCHAR2(50), reservation_count NUMBER)';
        DBMS_OUTPUT.PUT_LINE('Dočasná tabulka vytvořena');
        
        EXECUTE IMMEDIATE 'DROP TABLE temp_reservation_stats';
        DBMS_OUTPUT.PUT_LINE('Dočasná tabulka smazána');
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Chyba při práci s dočasnou tabulkou: ' || SQLERRM);
    END;
    
    COMMIT;
END;
/

-- =============================================
-- 4. ALTER COMPILE DEMO
-- =============================================

-- Vytvoření testovacích objektů pro hotel
CREATE OR REPLACE FUNCTION calc_room_revenue(p_room_id NUMBER) RETURN NUMBER AS
    v_revenue NUMBER;
BEGIN
    SELECT NVL(SUM(accommodation_price), 0) INTO v_revenue
    FROM Reservation 
    WHERE room_id = p_room_id AND status = 'Checked-out';
    
    RETURN v_revenue;
END;
/

CREATE OR REPLACE PROCEDURE show_guest_info(p_guest_id NUMBER) AS
    v_firstname VARCHAR2(100);
    v_lastname VARCHAR2(100);
    v_guest_type VARCHAR2(50);
BEGIN
    SELECT firstname, lastname, guest_type 
    INTO v_firstname, v_lastname, v_guest_type
    FROM Guest WHERE guest_id = p_guest_id;
    
    DBMS_OUTPUT.PUT_LINE('Host: ' || v_firstname || ' ' || v_lastname || ' (' || v_guest_type || ')');
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('Host s ID ' || p_guest_id || ' nenalezen');
END;
/

CREATE OR REPLACE PACKAGE hotel_management AS
    FUNCTION get_guest_count RETURN NUMBER;
    FUNCTION get_room_occupancy RETURN NUMBER;
    PROCEDURE show_reservation_summary(p_guest_id NUMBER);
END;
/

CREATE OR REPLACE PACKAGE BODY hotel_management AS
    FUNCTION get_guest_count RETURN NUMBER IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count FROM Guest;
        RETURN v_count;
    END;
    
    FUNCTION get_room_occupancy RETURN NUMBER IS
        v_occupied NUMBER;
        v_total NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_total FROM Room;
        SELECT COUNT(*) INTO v_occupied FROM Room WHERE is_occupied = 1;
        
        RETURN CASE WHEN v_total > 0 THEN ROUND((v_occupied / v_total) * 100, 2) ELSE 0 END;
    END;
    
    PROCEDURE show_reservation_summary(p_guest_id NUMBER) IS
        v_count NUMBER;
        v_total_spent NUMBER;
    BEGIN
        SELECT COUNT(*), NVL(SUM(accommodation_price), 0)
        INTO v_count, v_total_spent
        FROM Reservation 
        WHERE guest_id = p_guest_id;
        
        DBMS_OUTPUT.PUT_LINE('Host ID ' || p_guest_id || ': ' || v_count || ' rezervací, celkem ' || v_total_spent || ' Kč');
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            DBMS_OUTPUT.PUT_LINE('Žádné rezervace pro hosta ID ' || p_guest_id);
    END;
END;
/

-- Demonstrace ALTER COMPILE
CREATE OR REPLACE PROCEDURE demo_alter_compile AS
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO ALTER COMPILE ===');
    
    -- Kompilace funkce
    EXECUTE IMMEDIATE 'ALTER FUNCTION calc_room_revenue COMPILE';
    DBMS_OUTPUT.PUT_LINE('Funkce calc_room_revenue zkompilována');
    
    -- Kompilace procedury
    EXECUTE IMMEDIATE 'ALTER PROCEDURE show_guest_info COMPILE';
    DBMS_OUTPUT.PUT_LINE('Procedura show_guest_info zkompilována');
    
    -- Kompilace package specification
    EXECUTE IMMEDIATE 'ALTER PACKAGE hotel_management COMPILE SPECIFICATION';
    DBMS_OUTPUT.PUT_LINE('Package specification zkompilována');
    
    -- Kompilace package body
    EXECUTE IMMEDIATE 'ALTER PACKAGE hotel_management COMPILE BODY';
    DBMS_OUTPUT.PUT_LINE('Package body zkompilováno');
    
    -- Kompilace celého package
    EXECUTE IMMEDIATE 'ALTER PACKAGE hotel_management COMPILE';
    DBMS_OUTPUT.PUT_LINE('Celý package zkompilován');
    
END;
/

-- =============================================
-- 5. OPTIMALIZACE VÝKONU
-- =============================================

-- A) NOCOPY optimalizace
CREATE OR REPLACE PROCEDURE demo_nocopy AS
    TYPE t_guest_array IS TABLE OF Guest%ROWTYPE INDEX BY PLS_INTEGER;
    
    -- Bez NOCOPY - kopíruje celé pole
    PROCEDURE process_without_nocopy(p_guests IN t_guest_array) AS
        v_start_time NUMBER;
    BEGIN
        v_start_time := DBMS_UTILITY.GET_TIME;
        -- Simulace zpracování
        FOR i IN 1..p_guests.COUNT LOOP
            NULL; -- nějaké zpracování
        END LOOP;
        DBMS_OUTPUT.PUT_LINE('Bez NOCOPY: ' || (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    END;
    
    -- S NOCOPY - předává referenci
    PROCEDURE process_with_nocopy(p_guests IN NOCOPY t_guest_array) AS
        v_start_time NUMBER;
    BEGIN
        v_start_time := DBMS_UTILITY.GET_TIME;
        -- Simulace zpracování
        FOR i IN 1..p_guests.COUNT LOOP
            NULL; -- nějaké zpracování
        END LOOP;
        DBMS_OUTPUT.PUT_LINE('S NOCOPY: ' || (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    END;
    
    v_guests t_guest_array;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO NOCOPY ===');
    
    -- Naplnění pole daty
    SELECT * BULK COLLECT INTO v_guests FROM Guest WHERE ROWNUM <= 50;
    
    process_without_nocopy(v_guests);
    process_with_nocopy(v_guests);
END;
/

-- B) DETERMINISTIC optimalizace
CREATE OR REPLACE FUNCTION calculate_loyalty_points(p_guest_type VARCHAR2) 
RETURN NUMBER DETERMINISTIC AS
BEGIN
    RETURN CASE p_guest_type
        WHEN 'VIP' THEN 100
        WHEN 'Premium' THEN 50
        WHEN 'Business' THEN 30
        ELSE 10
    END;
END;
/

-- C) FORALL a BULK COLLECT optimalizace
CREATE OR REPLACE PROCEDURE demo_bulk_operations AS
    TYPE t_id_array IS TABLE OF NUMBER INDEX BY PLS_INTEGER;
    TYPE t_price_array IS TABLE OF NUMBER INDEX BY PLS_INTEGER;
    
    v_reservation_ids t_id_array;
    v_old_prices t_price_array;
    v_new_prices t_price_array;
    v_start_time NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO BULK OPERATIONS ===');
    
    -- BULK COLLECT - rychlé načítání dat
    v_start_time := DBMS_UTILITY.GET_TIME;
    
    SELECT reservation_id, accommodation_price
    BULK COLLECT INTO v_reservation_ids, v_old_prices
    FROM Reservation 
    WHERE status = 'Confirmed' AND ROWNUM <= 20;
    
    DBMS_OUTPUT.PUT_LINE('BULK COLLECT načetl ' || v_reservation_ids.COUNT || ' záznamů za: ' || 
                        (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    
    -- Úprava cen (5% sleva)
    FOR i IN 1..v_old_prices.COUNT LOOP
        v_new_prices(i) := v_old_prices(i) * 0.95;
    END LOOP;
    
    -- FORALL - rychlá aktualizace
    v_start_time := DBMS_UTILITY.GET_TIME;
    
    FORALL i IN 1..v_reservation_ids.COUNT
        UPDATE Reservation 
        SET accommodation_price = v_new_prices(i) 
        WHERE reservation_id = v_reservation_ids(i);
    
    DBMS_OUTPUT.PUT_LINE('FORALL aktualizoval ' || SQL%ROWCOUNT || ' záznamů za: ' || 
                        (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    
    COMMIT;
END;
/

-- D) Optimalizace s RETURNING klauzulí
CREATE OR REPLACE PROCEDURE demo_returning_clause AS
    TYPE t_id_array IS TABLE OF NUMBER INDEX BY PLS_INTEGER;
    TYPE t_price_array IS TABLE OF NUMBER INDEX BY PLS_INTEGER;
    
    v_guest_ids t_id_array;
    v_old_types t_price_array;
    v_new_ids t_id_array;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO RETURNING CLAUSE ===');
    
    -- Načtení ID hostů pro upgrade
    SELECT guest_id 
    BULK COLLECT INTO v_guest_ids
    FROM Guest 
    WHERE guest_type = 'Regular' AND ROWNUM <= 5;
    
    -- Upgrade hostů s RETURNING - vrací ID v jednom kroku
    FORALL i IN 1..v_guest_ids.COUNT
        UPDATE Guest 
        SET guest_type = 'Premium'
        WHERE guest_id = v_guest_ids(i)
        RETURNING guest_id BULK COLLECT INTO v_new_ids;
    
    -- Zobrazení výsledků
    FOR i IN 1..v_new_ids.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE('Host ID ' || v_new_ids(i) || ' upgradován na Premium');
    END LOOP;
    
    COMMIT;
END;
/

-- E) Porovnání výkonu FETCH vs BULK COLLECT
CREATE OR REPLACE PROCEDURE demo_fetch_vs_bulk AS
    CURSOR c_reservations IS 
        SELECT reservation_id, guest_id, accommodation_price 
        FROM Reservation 
        WHERE status IN ('Confirmed', 'Checked-in');
    
    TYPE t_res_array IS TABLE OF c_reservations%ROWTYPE;
    v_reservations t_res_array;
    v_res_rec c_reservations%ROWTYPE;
    v_start_time NUMBER;
    v_count NUMBER := 0;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO FETCH VS BULK COLLECT ===');
    
    -- 1. Klasický FETCH
    v_start_time := DBMS_UTILITY.GET_TIME;
    v_count := 0;
    
    OPEN c_reservations;
    LOOP
        FETCH c_reservations INTO v_res_rec;
        EXIT WHEN c_reservations%NOTFOUND;
        v_count := v_count + 1;
        -- Simulace zpracování
        NULL;
    END LOOP;
    CLOSE c_reservations;
    
    DBMS_OUTPUT.PUT_LINE('Klasický FETCH: ' || v_count || ' záznamů za ' || 
                        (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    
    -- 2. BULK COLLECT
    v_start_time := DBMS_UTILITY.GET_TIME;
    
    OPEN c_reservations;
    FETCH c_reservations BULK COLLECT INTO v_reservations;
    CLOSE c_reservations;
    
    -- Zpracování
    FOR i IN 1..v_reservations.COUNT LOOP
        -- Simulace zpracování
        NULL;
    END LOOP;
    
    DBMS_OUTPUT.PUT_LINE('BULK COLLECT: ' || v_reservations.COUNT || ' záznamů za ' || 
                        (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
END;
/

-- =============================================
-- 6. SPUŠTĚNÍ VŠECH DEMO PROCEDUR
-- =============================================

BEGIN
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    DBMS_OUTPUT.PUT_LINE('SPOUŠTĚNÍ VŠECH DEMO PROCEDUR - LEKCE 12 (HOTEL MANAGEMENT)');
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    
    -- Vytvoření testovacích dat
    create_test_data;
    DBMS_OUTPUT.PUT_LINE('');
    
    -- 1. DBMS_SQL
    demo_dbms_sql;
    DBMS_OUTPUT.PUT_LINE('');
    
    -- 2. EXECUTE IMMEDIATE
    demo_execute_immediate;
    DBMS_OUTPUT.PUT_LINE('');
    
    -- 3. ALTER COMPILE
    demo_alter_compile;
    DBMS_OUTPUT.PUT_LINE('');
    
    -- 4. Optimalizace výkonu
    demo_nocopy;
    DBMS_OUTPUT.PUT_LINE('');
    
    demo_bulk_operations;
    DBMS_OUTPUT.PUT_LINE('');
    
    demo_returning_clause;
    DBMS_OUTPUT.PUT_LINE('');
    
    demo_fetch_vs_bulk;
    DBMS_OUTPUT.PUT_LINE('');
    
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    DBMS_OUTPUT.PUT_LINE('VŠECHNY DEMO PROCEDURY DOKONČENY');
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
END;
/

