-- Vytvořte PACKAGE, s dříve implementovaných procedur a funkcí PACKAGE (hlavičku), PACKAGE BODY (tělo)

CREATE OR REPLACE PACKAGE guest_pkg AS
    FUNCTION guest_fullname (p_guest_id IN GUEST.GUEST_ID%TYPE) RETURN VARCHAR2;
    PROCEDURE proc_a;
    PROCEDURE proc_b;
END guest_pkg;
/

CREATE OR REPLACE PACKAGE BODY guest_pkg AS
    FUNCTION guest_fullname (p_guest_id IN GUEST.GUEST_ID%TYPE) RETURN VARCHAR2 IS
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

    PROCEDURE proc_a AS
    BEGIN
        DBMS_OUTPUT.PUT_LINE('In Procedure A');
        proc_b;  -- Call Procedure B
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Error in Procedure A: ' || SQLERRM);
    END;

    PROCEDURE proc_b AS
    BEGIN
        DBMS_OUTPUT.PUT_LINE('In Procedure B');
        RAISE_APPLICATION_ERROR(-20002, 'An error occurred in Procedure B');
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Error in Procedure B: ' || SQLERRM);
            RAISE;
    END;
END guest_pkg;


-- Přetěžování podprogramů – definujte PACKAGE, kde bude mít jedna procedura
-- (funkce) několik interpretací (provádění), podle počtu a typu vstupních parametrů
-- Implementuje FUNKCI, která bude mít libovolný počet textových parametrů, které
-- následně spojí v opačném pořadí dohromady, nebo vymyslete jinou alternativní
-- funkci (proceduru)
CREATE OR REPLACE PACKAGE string_pkg AS
    FUNCTION reverse_concat(p_str1 IN VARCHAR2) RETURN VARCHAR2;
    FUNCTION reverse_concat(p_str1 IN VARCHAR2, p_str2 IN VARCHAR2) RETURN VARCHAR2;
    FUNCTION reverse_concat(p_str1 IN VARCHAR2, p_str2 IN VARCHAR2, p_str3 IN VARCHAR2) RETURN VARCHAR2;
END string_pkg;

