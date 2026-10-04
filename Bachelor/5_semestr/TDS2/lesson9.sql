-- // Vytvořte funkci s návratovou hodnotou CREATE OR REPLACE FUNCTION název funkce
-- (param,…) RETURN datový typ IS …

CREATE OR REPLACE FUNCTION guest_fullname (
    p_guest_id IN GUEST.GUEST_ID%TYPE
) RETURN VARCHAR2 IS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
    v_fullname  VARCHAR2(200);
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = p_guest_id;

    v_fullname := v_firstname || ' ' || v_lastname;
    RETURN v_fullname;
END;

CREATE OR REPLACE FUNCTION bool_to_string(p_bool IN BOOLEAN) RETURN VARCHAR2 IS
BEGIN
    RETURN CASE 
        WHEN p_bool IS NULL THEN 'NULL'
        WHEN p_bool THEN 'TRUE'
        ELSE 'FALSE'
    END;
END;

-- Použijte vámi definovanou funkci přímo v příkazu jazyka SQL, v sekce SELECT, WHERE, GROUP BY, ORDER BY

SELECT guest_fullname(GUEST_ID) AS FULLNAME
FROM GUEST
WHERE GUEST_ID IN (1, 2, 3, 4, 5);

-- Prohlédněte si datový slovník s vašimi zdroji, USER_TABLES, USER_INDEXES, USER_SOURCES, USER_OBJECTS
SELECT OBJECT_NAME, OBJECT_TYPE, CREATED, LAST_DDL_TIME
FROM USER_OBJECTS
WHERE OBJECT_TYPE IN ('PROCEDURE', 'FUNCTION')
ORDER BY LAST_DDL_TIME DESC;

-- Napište dvě vzájemně se volající procedury s vyvoláním výjimek
CREATE OR REPLACE PROCEDURE proc_a;
CREATE OR REPLACE PROCEDURE proc_a AS
BEGIN
    DBMS_OUTPUT.PUT_LINE('In Procedure A');
    proc_b;  -- Call Procedure B
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error in Procedure A: ' || SQLERRM);
END;

CREATE OR REPLACE PROCEDURE proc_b AS
BEGIN
    DBMS_OUTPUT.PUT_LINE('In Procedure B');
    RAISE_APPLICATION_ERROR(-20002, 'An error occurred in Procedure B');
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error in Procedure B: ' || SQLERRM);
        RAISE;
END;

BEGIN
    proc_a;
END;


-- Ověřte oprávnění k manipulaci s jednotlivými objekty TABULKy, SEKVENCe, POHLEDy, PROCEDURy

