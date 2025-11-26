-- Ukol 1
-- anonymni funkce, declare, block a vyjimka
DECLARE
    payment_id NUMBER := 3;
    payment_price NUMBER;
    payment_discount FLOAT := 0.1;
    payment_price_discounted NUMBER;

BEGIN
    SELECT TOTAL_EXPENSES + TOTAL_ACCOMMODATION
    INTO payment_price
    FROM PAYMENT
    WHERE PAYMENT_ID = 2;


    IF payment_price IS NOT NULL THEN
        payment_price_discounted := payment_price * (1 - payment_discount);
        DBMS_OUTPUT.PUT_LINE('Discount for payment ' || payment_price ||
                             ' is: ' || payment_price_discounted);
    ELSE
        DBMS_OUTPUT.PUT_LINE('No payment data for payment id: ' || payment_id || ' found');
    END IF;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No data found for payment ' || payment_id);
    WHEN TOO_MANY_ROWS THEN
        DBMS_OUTPUT.PUT_LINE('More than one row found for payment ' || payment_id);
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred: ' || SQLERRM);
END;


-- Lekce 2

DECLARE
    -- 1. Proměnné základních typů (String, Integer, Date, BOOLEAN)
    myString VARCHAR2(100) := 'Hello, My Small World!';
    myInteger INTEGER := 10;
    myDate DATE := SYSDATE;
    myBoolean BOOLEAN := TRUE;

    -- 2. Definování konstanty
    CONSTANT_VALUE CONSTANT NUMBER := 100;

    -- 3. Proměnná s defaultní hodnotou
    defaultValue NUMBER DEFAULT 50;

    -- 4. Proměnné s použitím %TYPE (TOTO VÁM CHYBĚLO!)
    salary NUMBER := 80000;
    employeeSalary salary%TYPE;  -- Převezme datový typ z proměnné salary
    bonus salary%TYPE;          -- Převezme datový typ z proměnné salary
    bonusPercentage NUMBER(3, 2) := 0.10;
    currentPercentage bonusPercentage%TYPE := 0.15; -- Převezme typ z bonusPercentage

    -- Nested procedure (vnořená procedura)
    PROCEDURE CalculateBonus IS
        calculatedBonus NUMBER;
    BEGIN
        calculatedBonus := salary * bonusPercentage;
        DBMS_OUTPUT.PUT_LINE('Calculated Bonus: ' || calculatedBonus);
    END CalculateBonus;

BEGIN
    -- 5. Implicitní/explicitní konverze datových typů
    myInteger := TO_NUMBER('20');  -- Explicitní konverze
    employeeSalary := salary;      -- Implicitní konverze (stejný typ díky %TYPE)
    bonus := salary * 0.2;        -- Implicitní konverze

    -- 6. Výpis výsledků na konzolu (DBMS_OUTPUT.PUT_LINE)
    DBMS_OUTPUT.PUT_LINE('String: ' || myString);
    DBMS_OUTPUT.PUT_LINE('Integer: ' || myInteger);
    DBMS_OUTPUT.PUT_LINE('Date: ' || TO_CHAR(myDate, 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Boolean: ' || CASE WHEN myBoolean THEN 'TRUE' ELSE 'FALSE' END);
    DBMS_OUTPUT.PUT_LINE('Constant: ' || CONSTANT_VALUE);
    DBMS_OUTPUT.PUT_LINE('Default Value: ' || defaultValue);
    DBMS_OUTPUT.PUT_LINE('Employee Salary (%TYPE): ' || employeeSalary);
    DBMS_OUTPUT.PUT_LINE('Bonus (%TYPE): ' || bonus);
    DBMS_OUTPUT.PUT_LINE('Current Percentage (%TYPE): ' || currentPercentage);

    -- 7. Volání vnořené procedury
    CalculateBonus;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred: ' || SQLERRM);
END;

-- Ukol 3

-- 1. INSERT nového záznamu
BEGIN
    manage_service_price_history(
        p_sph_id => 999,
        new_price => 150.00,
        p_valid_to => TO_DATE('2024-12-31', 'YYYY-MM-DD'),
        p_service_id => 1,
        dell_flag => FALSE,
        new_flag => TRUE,
        merge_flag => FALSE
    );
END;
/

-- 2. UPDATE existujícího záznamu
BEGIN
    manage_service_price_history(
        p_sph_id => 1,
        new_price => 200.00,
        p_valid_to => TO_DATE('2024-12-31', 'YYYY-MM-DD'),
        p_service_id => 1,
        dell_flag => FALSE,
        new_flag => FALSE,
        merge_flag => FALSE
    );
END;

-- 3. DELETE záznamu
BEGIN
    manage_service_price_history(
        p_sph_id => 999,
        new_price => 0,
        p_valid_to => NULL,
        p_service_id => 0,
        dell_flag => TRUE,
        new_flag => FALSE,
        merge_flag => FALSE
    );
END;
/

-- 4. MERGE operace
BEGIN
    manage_service_price_history(
        p_sph_id => 1,
        new_price => 175.00,
        p_valid_to => TO_DATE('2024-12-31', 'YYYY-MM-DD'),
        p_service_id => 1,
        dell_flag => FALSE,
        new_flag => FALSE,
        merge_flag => TRUE
    );
END;

-- Ukol 4
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 4 - TESTING PROCEDURES ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. IF-THEN-ELSE a IF-ELSEIF-ELSE
    DBMS_OUTPUT.PUT_LINE('1. Testing IF-THEN-ELSE and IF-ELSEIF-ELSE:');
    FEEDBACK_RATING(1);
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. CASE-WHEN-ELSE (s proměnnou za CASE) jako výsledek přiřazení
    DBMS_OUTPUT.PUT_LINE('2. Testing CASE-WHEN-ELSE (assignment to variable):');
    GUEST_TYPE(1);
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. CASE-WHEN-ELSE s podmínkou za WHEN
    DBMS_OUTPUT.PUT_LINE('3. Testing CASE-WHEN-ELSE (condition after WHEN):');
    ROOM_BEDS(1);
    DBMS_OUTPUT.PUT_LINE('');

    -- 4. Logické operátory AND, OR, NOT s TRUE, FALSE, NULL
    DBMS_OUTPUT.PUT_LINE('4. Testing logical operators (AND, OR, NOT):');
    CHECK_LOGICAL_OPERATORS(TRUE, FALSE);
    DBMS_OUTPUT.PUT_LINE('');

    -- 5. LOOP – EXIT [WHEN] – END LOOP
    DBMS_OUTPUT.PUT_LINE('5. Testing LOOP with EXIT WHEN:');
    SIMPLE_COUNTDOWN(5);
    DBMS_OUTPUT.PUT_LINE('');

    -- 6. WHILE – podmínka LOOP – END LOOP
    DBMS_OUTPUT.PUT_LINE('6. Testing WHILE LOOP:');
    WHILE_COUNTDOWN(5);
    DBMS_OUTPUT.PUT_LINE('');

    -- 7. FOR počítadlo IN spodní..horní LOOP – END LOOP
    DBMS_OUTPUT.PUT_LINE('7. Testing FOR LOOP:');
    FOR_COUNTDOWN(5);
    DBMS_OUTPUT.PUT_LINE('');

    -- 8. Cyklus s reverzním čítačem REVERSE
    DBMS_OUTPUT.PUT_LINE('8. Testing REVERSE FOR LOOP:');
    REVERSE_COUNTDOWN(5);
    DBMS_OUTPUT.PUT_LINE('');

    -- 9. Zanořené cykly s ukončením EXIT a návěštími
    DBMS_OUTPUT.PUT_LINE('9. Testing nested loops with EXIT and labels:');
    NESTED_LOOPS_EXIT_LABELS;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL TESTS COMPLETED ===');
END;


-- Ukol 5

BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 5 - TESTING CURSORS ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. Explicitní kurzor s OPEN-FETCH-CLOSE
    DBMS_OUTPUT.PUT_LINE('1. Testing explicit cursor (OPEN-FETCH-CLOSE):');
    CHECK_GUEST;
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Kurzor s %ROWTYPE a flags (%ISOPEN, %NOTFOUND, %FOUND, %ROWCOUNT)
    DBMS_OUTPUT.PUT_LINE('2. Testing cursor with %ROWTYPE and flags:');
    CHECK_GUEST_RECORD;
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Kurzor FOR LOOP s EXIT WHEN
    DBMS_OUTPUT.PUT_LINE('3. Testing cursor FOR LOOP:');
    CHECK_GUEST_FOR_LOOP;
    DBMS_OUTPUT.PUT_LINE('');

    -- 4. Kurzor s parametry
    DBMS_OUTPUT.PUT_LINE('4. Testing cursor with parameters:');
    CHECK_GUEST_TYPE('CONFIRMED');  -- Zkuste různé statusy: 'CONFIRMED', 'PENDING', 'CANCELLED'
    DBMS_OUTPUT.PUT_LINE('');

    -- 5. Kurzor pro UPDATE s WAIT/NOWAIT
    DBMS_OUTPUT.PUT_LINE('5. Testing cursor for UPDATE with WAIT:');
    UPDATE_GUEST_TYPE;
    DBMS_OUTPUT.PUT_LINE('');

    -- 6. Vnořené kurzory
    DBMS_OUTPUT.PUT_LINE('6. Testing nested cursors:');
    CHECK_GUEST_NESTED;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL CURSOR TESTS COMPLETED ===');
END;


-- Ukol 6
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 6 - TESTING RECORDS AND COLLECTIONS ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. %ROWTYPE record z tabulky
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('1. Testing %ROWTYPE record from table:');
    guest_info_rowtype;
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Vlastní record type s TYPE IS RECORD
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('2. Testing custom record type (TYPE IS RECORD):');
    guest_info;
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Tabulka (pole) INDEXED BY PLS_INTEGER
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('3. Testing TABLE INDEXED BY PLS_INTEGER:');
    guest_info_table;
    DBMS_OUTPUT.PUT_LINE('');

    -- 4. Tabulka vlastních recordů INDEXED BY PLS_INTEGER
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('4. Testing TABLE OF custom RECORDS:');
    guest_info_table_of_records;
    DBMS_OUTPUT.PUT_LINE('');

    -- 5. INDEXED BY BINARY_INTEGER
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('5. Testing INDEXED BY BINARY_INTEGER:');
    guest_info_indexed_by_binary_integer;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL RECORDS AND COLLECTIONS TESTS COMPLETED ===');
END;

-- Ukol  7
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 7 - TESTING EXCEPTION HANDLING ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. Základní ošetřování výjimek (NO_DATA_FOUND, TOO_MANY_ROWS, OTHERS)
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('1. Testing basic exception handling:');
    BASIC_GUSET_INFO();
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Zachytávání Oracle server výjimek (SQLCODE)
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('2. Testing Oracle server exceptions:');
    guest_info_exceptions;
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Vlastní název výjimky s PRAGMA EXCEPTION_INIT
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('3. Testing custom exception with PRAGMA:');
    guest_info_custom_exception;
    DBMS_OUTPUT.PUT_LINE('');

    -- 4. Výpis SQLCODE a SQLERRM
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('4. Testing SQLCODE and SQLERRM:');
    guest_info_custom_exception_msg;
    DBMS_OUTPUT.PUT_LINE('');

    -- 5. RAISE_APPLICATION_ERROR
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('5. Testing RAISE_APPLICATION_ERROR:');
    guest_info_custom_exception_msg2;
    DBMS_OUTPUT.PUT_LINE('');

    -- 6. Vnořené procedury s výjimkami
    DBMS_OUTPUT.PUT_LINE('==========================================');
    DBMS_OUTPUT.PUT_LINE('6. Testing nested procedure exceptions:');
    guest_info_nested_exception;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL EXCEPTION TESTS COMPLETED ===');
END;



-- Ukol 8
-- Running script for testing Procedures and Parameters
DECLARE
    v_fullname VARCHAR2(200);
    v_guest_id NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 8 - TESTING PROCEDURES AND PARAMETERS ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. Jednoduchá procedura bez parametrů
    DBMS_OUTPUT.PUT_LINE('1. Testing simple procedure (no parameters):');
    guest_info_simple;
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Procedura s IN a OUT parametry
    DBMS_OUTPUT.PUT_LINE('2. Testing IN and OUT parameters:');
    guest_info_param(p_guest_id => 1, p_fullname => v_fullname);
    DBMS_OUTPUT.PUT_LINE('Returned fullname: ' || v_fullname);
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Test OUT parametru (co se stane při pokusu o čtení)
    DBMS_OUTPUT.PUT_LINE('3. Testing OUT parameter behavior:');
    guest_info_param_out(p_guest_id => 1, p_fullname => v_fullname);
    DBMS_OUTPUT.PUT_LINE('Returned fullname: ' || v_fullname);
    DBMS_OUTPUT.PUT_LINE('');

    -- 4. Přiřazování parametrů podle názvu + DEFAULT hodnoty
    DBMS_OUTPUT.PUT_LINE('4. Testing named parameters and DEFAULT values:');

    -- Volání s DEFAULT hodnotou (bez p_guest_id)
    guest_info_param_named(p_fullname => v_fullname);
    DBMS_OUTPUT.PUT_LINE('With DEFAULT guest_id=1: ' || v_fullname);

    -- Volání podle názvu parametrů
    guest_info_param_named(p_guest_id => 2, p_fullname => v_fullname);
    DBMS_OUTPUT.PUT_LINE('With guest_id=2: ' || v_fullname);

    -- Volání podle pořadí parametrů
    v_guest_id := 3;
    guest_info_param_named(v_guest_id, v_fullname);
    DBMS_OUTPUT.PUT_LINE('With guest_id=3 (by position): ' || v_fullname);
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL PROCEDURE TESTS COMPLETED ===');
END;

-- Ukol 9
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 9 - TESTING FUNCTIONS AND ADVANCED TOPICS ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. Test funkce přímo v PL/SQL bloku
    DBMS_OUTPUT.PUT_LINE('1. Testing function in PL/SQL block:');
    DBMS_OUTPUT.PUT_LINE('Guest 1: ' || guest_fullname(1));
    DBMS_OUTPUT.PUT_LINE('Guest 2: ' || guest_fullname(2));
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Test boolean funkce
    DBMS_OUTPUT.PUT_LINE('2. Testing boolean function:');
    DBMS_OUTPUT.PUT_LINE('TRUE: ' || bool_to_string(TRUE));
    DBMS_OUTPUT.PUT_LINE('FALSE: ' || bool_to_string(FALSE));
    DBMS_OUTPUT.PUT_LINE('NULL: ' || bool_to_string(NULL));
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Test vzájemně se volajících procedur
    DBMS_OUTPUT.PUT_LINE('3. Testing mutually calling procedures with exceptions:');
    proc_a;
    DBMS_OUTPUT.PUT_LINE('');

    DBMS_OUTPUT.PUT_LINE('=== ALL FUNCTION TESTS COMPLETED ===');
END;


-- Ukol 10
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 10 - PACKAGES & OVERLOADING DEMO ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- 1. Test guest_pkg
    DBMS_OUTPUT.PUT_LINE('1. GUEST PACKAGE:');
    DBMS_OUTPUT.PUT_LINE('Guest 1: ' || guest_pkg.guest_fullname(1));
    DBMS_OUTPUT.PUT_LINE('Guest 2: ' || guest_pkg.guest_fullname(2));
    DBMS_OUTPUT.PUT_LINE('');

    -- Test procedur s výjimkami
    DBMS_OUTPUT.PUT_LINE('Testing exception handling:');
    guest_pkg.proc_a; -- Volá proc_b, který vyhodí chybu
    DBMS_OUTPUT.PUT_LINE('');

    -- 2. Test string_pkg OVERLOADING
    DBMS_OUTPUT.PUT_LINE('2. OVERLOADING DEMO:');
    DBMS_OUTPUT.PUT_LINE('1 param: "' || string_pkg.reverse_concat('Hello') || '"');
    DBMS_OUTPUT.PUT_LINE('2 params: "' || string_pkg.reverse_concat('Hello', 'World') || '"');
    DBMS_OUTPUT.PUT_LINE('3 params: "' || string_pkg.reverse_concat('A', 'B', 'C') || '"');
    DBMS_OUTPUT.PUT_LINE('');

    -- 3. Více příkladů overloadingu
    DBMS_OUTPUT.PUT_LINE('3. MORE OVERLOADING EXAMPLES:');
    DBMS_OUTPUT.PUT_LINE('Names: "' || string_pkg.reverse_concat('John', 'Doe') || '"');
    DBMS_OUTPUT.PUT_LINE('Colors: "' || string_pkg.reverse_concat('Red', 'Green', 'Blue') || '"');

    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== DEMO COMPLETED ===');
    DBMS_OUTPUT.PUT_LINE('✓ Packages created');
    DBMS_OUTPUT.PUT_LINE('✓ Function overloading works');
    DBMS_OUTPUT.PUT_LINE('✓ Exception handling demonstrated');

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
END;



-- Ukol 11
BEGIN
    DBMS_OUTPUT.PUT_LINE('=== LESSON 11 - PERSISTENT PACKAGE STATE DEMO ===');
    DBMS_OUTPUT.PUT_LINE('');

    -- Blok 1: Nastavení hodnoty
    DBMS_OUTPUT.PUT_LINE('BLOCK 1: Setting value');
    global_package.set_global_variable('Hello from Block 1');
    DBMS_OUTPUT.PUT_LINE('Current value: ' || global_package.get_global_variable);
    DBMS_OUTPUT.PUT_LINE('');

    -- Blok 2: Čtení hodnoty (zůstala!)
    DBMS_OUTPUT.PUT_LINE('BLOCK 2: Reading value - should persist!');
    DBMS_OUTPUT.PUT_LINE('Persisted value: ' || global_package.get_global_variable);

    -- Změna hodnoty
    global_package.set_global_variable('Updated in Block 2');
    DBMS_OUTPUT.PUT_LINE('New value: ' || global_package.get_global_variable);
    DBMS_OUTPUT.PUT_LINE('');

    -- Blok 3: Finální test
    DBMS_OUTPUT.PUT_LINE('BLOCK 3: Final test');
    DBMS_OUTPUT.PUT_LINE('Final persisted value: ' || global_package.get_global_variable);

    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== PERSISTENCE DEMONSTRATED! ===');
    DBMS_OUTPUT.PUT_LINE('Package variables retain values between PL/SQL blocks');
END;

-- Ukol 12
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


-- Ukol 13
BEGIN
    test_triggers;
END;

-- Ukol 14

SELECT name, referenced_name, referenced_type
FROM user_dependencies
WHERE name = 'PROC_B';

SELECT LPAD(' ', 2*(LEVEL-1)) || name AS object_name,
       type,
       referenced_name,
       referenced_type
FROM user_dependencies
START WITH name = 'PROC_B'
CONNECT BY PRIOR referenced_name = name;

-- 3) TIMESTAMP a SIGNATURE ukázka
SELECT *
FROM user_dependencies
WHERE name = 'PROC_A';

SELECT *
FROM user_dependencies
WHERE name = 'PROC_B';

-- Ukol 15

CREATE OR REPLACE PROCEDURE proc_a IS
BEGIN
  DBMS_OUTPUT.PUT_LINE('Procedura A');
END;
/

CREATE OR REPLACE PROCEDURE proc_b IS
BEGIN
  proc_a;
  DBMS_OUTPUT.PUT_LINE('Procedura B');
END;
/

SELECT name, referenced_name, referenced_type
FROM user_dependencies
WHERE name = 'PROC_B';

SELECT LPAD(' ', 2*(LEVEL-1)) || name AS object_name,
       type,
       referenced_name,
       referenced_type
FROM user_dependencies
START WITH name = 'PROC_B'
CONNECT BY PRIOR referenced_name = name;

-- 3) TIMESTAMP a SIGNATURE ukázka
SELECT *
FROM user_dependencies
WHERE name = 'PROC_A';

SELECT *
FROM user_dependencies
WHERE name = 'PROC_B';