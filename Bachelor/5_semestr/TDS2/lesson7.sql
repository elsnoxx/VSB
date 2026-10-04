-- Ošetřování výjimek EXEPTION WHEN … THEN s WHEN OTHERS THEN
CREATE OR REPLACE PROCEDURE basic_guset_info AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = 100;

    DBMS_OUTPUT.PUT_LINE('Guest: ' || v_firstname || ' ' || v_lastname);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No guest with such ID');
    WHEN TOO_MANY_ROWS THEN
        DBMS_OUTPUT.PUT_LINE('More than one guest with such ID');
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Some other error: ' || SQLERRM);
END;

-- Zachytávání výjimek Oracle serveru předdefinované (konstanty), nepředefinované (číslované chyby)
CREATE OR REPLACE PROCEDURE guest_info_exceptions AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
    v_err_msg   VARCHAR2(200);
    v_err_code  NUMBER;
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = 100;

    DBMS_OUTPUT.PUT_LINE('Guest: ' || v_firstname || ' ' || v_lastname);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No data found.');
    WHEN OTHERS THEN
        IF SQLCODE = -942 THEN
            DBMS_OUTPUT.PUT_LINE('Table not found.');
    ELSE
        DBMS_OUTPUT.PUT_LINE('An error occurred: ' || SQLERRM);
    END IF;
END;

-- Deklarace vlastního názvu výjimky DECLARATION název výjimky EXCEPTION, PRAGMA EXCEPTION INIT …
CREATE OR REPLACE PROCEDURE guest_info_custom_exception AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;

    -- Deklarace vlastního názvu výjimky
    e_guest_not_found EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_guest_not_found, -20001);
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = 10;

    DBMS_OUTPUT.PUT_LINE('Guest: ' || v_firstname || ' ' || v_lastname);
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RAISE e_guest_not_found;
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred: ' || SQLERRM);
END;


-- Výpis chybového stavu a komentáře k výjimce: SQLCODE, SQLERRM
CREATE OR REPLACE PROCEDURE guest_info_custom_exception_msg AS
BEGIN
    RAISE NO_DATA_FOUND;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error code: ' || SQLCODE);
        DBMS_OUTPUT.PUT_LINE('Error message: ' || SQLERRM);
END;

-- Uživatelsky definované výjimky RAISE_APPLICATION_ERROR s definovaným názve a
-- druhá varianta s číslem vlastní výjimky a komentářem přímo v těle procedury
CREATE OR REPLACE PROCEDURE guest_info_custom_exception_msg2 AS
BEGIN
    RAISE_APPLICATION_ERROR(-20001, 'Guest not found with such ID');
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error code: ' || SQLCODE);
        DBMS_OUTPUT.PUT_LINE('Error message: ' || SQLERRM);
END;

-- Vyvolání výjimky v zanořené proceduře a její ošetření v zanořené nebo v nadřazené
CREATE OR REPLACE PROCEDURE guest_info_nested_exception AS

    PROCEDURE inner_procedure IS
    BEGIN
        RAISE_APPLICATION_ERROR(-20001, 'Guest not found with such ID in inner procedure');
    EXCEPTION
        WHEN OTHERS THEN
            RAISE;
    END inner_procedure;

BEGIN
    inner_procedure;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Error code: ' || SQLCODE);
        DBMS_OUTPUT.PUT_LINE('Error message: ' || SQLERRM);
END guest_info_nested_exception;