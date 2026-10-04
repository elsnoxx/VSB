-- Lesson 12 - Zkrácená verze pro vaši DB
SET SERVEROUTPUT ON;

-- =============================================
-- A) DYNAMICKÉ SQL POMOCÍ DBMS_SQL
-- =============================================

CREATE OR REPLACE PROCEDURE execute_dynamic_sql(
    p_sql_query VARCHAR2,
    p_input_param NUMBER,
    p_output_param OUT NUMBER
) AS
    v_cursor INTEGER;
    v_result NUMBER;
BEGIN
    v_cursor := DBMS_SQL.OPEN_CURSOR;

    DBMS_SQL.PARSE(v_cursor, p_sql_query, DBMS_SQL.NATIVE);
    DBMS_SQL.BIND_VARIABLE(v_cursor, ':input_param', p_input_param);
    DBMS_SQL.DEFINE_COLUMN(v_cursor, 1, p_output_param);

    v_result := DBMS_SQL.EXECUTE(v_cursor);

    IF DBMS_SQL.FETCH_ROWS(v_cursor) > 0 THEN
        DBMS_SQL.COLUMN_VALUE(v_cursor, 1, p_output_param);
    END IF;

    DBMS_SQL.CLOSE_CURSOR(v_cursor);
EXCEPTION
    WHEN OTHERS THEN
        IF DBMS_SQL.IS_OPEN(v_cursor) THEN
            DBMS_SQL.CLOSE_CURSOR(v_cursor);
        END IF;
        DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
        RAISE;
END;
/

-- Procedura pro dynamické dotazy na Guest tabulku
CREATE OR REPLACE PROCEDURE demo_guest_dynamic_sql AS
    v_cursor_id INTEGER;
    v_sql VARCHAR2(1000);
    v_guest_id NUMBER;
    v_firstname VARCHAR2(100);
    v_lastname VARCHAR2(100);
    v_guest_type VARCHAR2(50);
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO DBMS_SQL S GUEST TABULKOU ===');
    
    v_cursor_id := DBMS_SQL.OPEN_CURSOR;
    
    v_sql := 'SELECT guest_id, firstname, lastname, guest_type FROM Guest WHERE guest_type = :type';
    DBMS_SQL.PARSE(v_cursor_id, v_sql, DBMS_SQL.NATIVE);
    DBMS_SQL.BIND_VARIABLE(v_cursor_id, ':type', 'VIP');
    
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 1, v_guest_id);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 2, v_firstname, 100);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 3, v_lastname, 100);
    DBMS_SQL.DEFINE_COLUMN(v_cursor_id, 4, v_guest_type, 50);
    
    DBMS_SQL.EXECUTE(v_cursor_id);
    
    DBMS_OUTPUT.PUT_LINE('VIP hosté:');
    WHILE DBMS_SQL.FETCH_ROWS(v_cursor_id) > 0 LOOP
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 1, v_guest_id);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 2, v_firstname);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 3, v_lastname);
        DBMS_SQL.COLUMN_VALUE(v_cursor_id, 4, v_guest_type);
        
        DBMS_OUTPUT.PUT_LINE(v_guest_id || ': ' || v_firstname || ' ' || v_lastname || ' (' || v_guest_type || ')');
    END LOOP;
    
    DBMS_SQL.CLOSE_CURSOR(v_cursor_id);
EXCEPTION
    WHEN OTHERS THEN
        IF DBMS_SQL.IS_OPEN(v_cursor_id) THEN
            DBMS_SQL.CLOSE_CURSOR(v_cursor_id);
        END IF;
        DBMS_OUTPUT.PUT_LINE('Chyba: ' || SQLERRM);
END;


/

-- =============================================
-- B) ALTER COMPILE DEMO
-- =============================================

-- Testovací procedura
CREATE OR REPLACE PROCEDURE my_procedure AS
BEGIN
    DBMS_OUTPUT.PUT_LINE('Hello from my_procedure');
END;
/

-- Testovací funkce pro hotel
CREATE OR REPLACE FUNCTION my_function RETURN NUMBER AS
BEGIN
    RETURN 42;
END;
/

-- Funkce pro počítání hostů
CREATE OR REPLACE FUNCTION count_guests_by_type(p_guest_type VARCHAR2) RETURN NUMBER AS
    v_count NUMBER;
BEGIN
    SELECT COUNT(*) INTO v_count 
    FROM Guest 
    WHERE guest_type = p_guest_type;
    
    RETURN v_count;
END;
/

-- Package pro hotel management
CREATE OR REPLACE PACKAGE compiled_package IS
    PROCEDURE my_procedure;
    FUNCTION get_guest_count RETURN NUMBER;
    PROCEDURE show_guest_stats;
END compiled_package;
/

CREATE OR REPLACE PACKAGE BODY compiled_package IS
    PROCEDURE my_procedure AS
    BEGIN
        DBMS_OUTPUT.PUT_LINE('Hello from compiled_package.my_procedure');
    END;
    
    FUNCTION get_guest_count RETURN NUMBER IS
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count FROM Guest;
        RETURN v_count;
    END;
    
    PROCEDURE show_guest_stats AS
    BEGIN
        FOR rec IN (SELECT guest_type, COUNT(*) as cnt FROM Guest GROUP BY guest_type) LOOP
            DBMS_OUTPUT.PUT_LINE(rec.guest_type || ': ' || rec.cnt || ' hostů');
        END LOOP;
    END;
END compiled_package;
/

-- =============================================
-- C) OPTIMALIZACE - BULK COLLECT DEMO
-- =============================================

CREATE OR REPLACE PROCEDURE demo_bulk_collect AS
    TYPE t_guest_array IS TABLE OF Guest%ROWTYPE;
    v_guests t_guest_array;
    v_start_time NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== DEMO BULK COLLECT ===');
    
    v_start_time := DBMS_UTILITY.GET_TIME;
    
    -- BULK COLLECT načtení všech hostů
    SELECT * BULK COLLECT INTO v_guests 
    FROM Guest 
    WHERE ROWNUM <= 20;
    
    DBMS_OUTPUT.PUT_LINE('BULK COLLECT načetl ' || v_guests.COUNT || ' hostů za: ' || 
                        (DBMS_UTILITY.GET_TIME - v_start_time) || ' centisekund');
    
    -- Zpracování dat
    FOR i IN 1..v_guests.COUNT LOOP
        IF MOD(i, 5) = 0 THEN  -- Každý 5. host
            DBMS_OUTPUT.PUT_LINE('Host ' || i || ': ' || v_guests(i).firstname || ' ' || v_guests(i).lastname);
        END IF;
    END LOOP;
END;
/

-- =============================================
-- D) SPUŠTĚNÍ TESTŮ
-- =============================================

BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 12 - DYNAMIC SQL AND COMPILATION ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- Test 1: Dynamické SQL
    DBMS_OUTPUT.PUT_LINE('1. Testing dynamic SQL:');
    DECLARE
        my_input_param NUMBER := 42;
        my_output_param NUMBER;
    BEGIN
        execute_dynamic_sql('SELECT :input_param * 2 FROM DUAL', my_input_param, my_output_param);
        DBMS_OUTPUT.PUT_LINE('Result: ' || my_output_param);
    END;
    DBMS_OUTPUT.PUT_LINE('');

    -- Test 2: Dynamické SQL s Guest tabulkou
    DBMS_OUTPUT.PUT_LINE('2. Testing dynamic SQL with Guest table:');
    demo_guest_dynamic_sql;
    DBMS_OUTPUT.PUT_LINE('');

    -- Test 3: ALTER COMPILE
    DBMS_OUTPUT.PUT_LINE('3. Testing ALTER COMPILE:');
    
    -- Kompilace procedury
    EXECUTE IMMEDIATE 'ALTER PROCEDURE my_procedure COMPILE';
    DBMS_OUTPUT.PUT_LINE('Procedura my_procedure zkompilována');
    my_procedure;
    
    -- Kompilace funkce
    EXECUTE IMMEDIATE 'ALTER FUNCTION my_function COMPILE';
    DBMS_OUTPUT.PUT_LINE('Funkce my_function zkompilována');
    DECLARE
        result NUMBER;
    BEGIN
        result := my_function;
        DBMS_OUTPUT.PUT_LINE('Result from my_function: ' || result);
    END;
    
    -- Kompilace package
    EXECUTE IMMEDIATE 'ALTER PACKAGE compiled_package COMPILE';
    DBMS_OUTPUT.PUT_LINE('Package compiled_package zkompilován');
    compiled_package.my_procedure;
    DBMS_OUTPUT.PUT_LINE('');

    -- Test 4: Hotel funkce
    DBMS_OUTPUT.PUT_LINE('4. Testing hotel functions:');
    DBMS_OUTPUT.PUT_LINE('Celkem hostů: ' || compiled_package.get_guest_count);
    DBMS_OUTPUT.PUT_LINE('VIP hostů: ' || count_guests_by_type('VIP'));
    DBMS_OUTPUT.PUT_LINE('Regular hostů: ' || count_guests_by_type('Regular'));
    compiled_package.show_guest_stats;
    DBMS_OUTPUT.PUT_LINE('');

    -- Test 5: BULK COLLECT
    DBMS_OUTPUT.PUT_LINE('5. Testing BULK COLLECT optimization:');
    demo_bulk_collect;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL TESTS COMPLETED ===');
    
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error in main test: ' || SQLERRM);
        ROLLBACK;
END;
/