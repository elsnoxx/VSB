-- Vytvoření procedury – i předchozí příkazy můžete tvořit jako zanořené procedury
CREATE OR REPLACE PROCEDURE guest_info_simple AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = 1;

    DBMS_OUTPUT.PUT_LINE('Guest: ' || v_firstname || ' ' || v_lastname);
END;


-- Použití parametrů u procedur IN, OUT, IN OUT
CREATE OR REPLACE PROCEDURE guest_info_param (
    p_guest_id IN  GUEST.GUEST_ID%TYPE,
    p_fullname OUT VARCHAR2
) AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = p_guest_id;

    p_fullname := v_firstname || ' ' || v_lastname;
    DBMS_OUTPUT.PUT_LINE('Guest: ' || p_fullname);
END;


-- Zkuste, co se stane, když vložíte nějakou hodnotu do parametru OUT a bude s ní chtít v těle procedury pracovat
CREATE OR REPLACE PROCEDURE guest_info_param_out (
    p_guest_id IN  GUEST.GUEST_ID%TYPE,
    p_fullname OUT VARCHAR2
) AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
BEGIN
    -- Toto způsobí chybu, protože OUT parametry nemohou být čteny, pouze zapisovány
    p_fullname := 'Initial value';  

    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = p_guest_id;

    p_fullname := v_firstname || ' ' || v_lastname;
    DBMS_OUTPUT.PUT_LINE('Guest: ' || p_fullname);
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred: ' || SQLERRM);
END;

-- Přiřazování parametrů pořadím, nebo přiřazením podle názvu název parametru => hodnota parametru, použití DEFAULT hodnoty u parametru procedury
CREATE OR REPLACE PROCEDURE guest_info_param_named (
    p_guest_id IN  GUEST.GUEST_ID%TYPE DEFAULT 1,
    p_fullname OUT VARCHAR2
) AS
    v_firstname GUEST.FIRSTNAME%TYPE;
    v_lastname  GUEST.LASTNAME%TYPE;
BEGIN
    SELECT FIRSTNAME, LASTNAME
    INTO v_firstname, v_lastname
    FROM GUEST
    WHERE GUEST_ID = p_guest_id;

    p_fullname := v_firstname || ' ' || v_lastname;
    DBMS_OUTPUT.PUT_LINE('Guest: ' || p_fullname);
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An error occurred: ' || SQLERRM);
END;