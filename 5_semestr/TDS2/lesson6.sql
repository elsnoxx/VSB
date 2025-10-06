

-- Implement user defined %ROWTYPE records taken from table

CREATE OR REPLACE PROCEDURE guest_info_rowtype AS
    v_guest GUEST%ROWTYPE;
BEGIN
    SELECT *
    INTO v_guest
    FROM GUEST
    WHERE GUEST_ID = 1;

    DBMS_OUTPUT.PUT_LINE(
        v_guest.FIRSTNAME || ' ' ||
        v_guest.LASTNAME  || ' lives in: ' ||
        v_guest.CITY      || ' and customer registrated from ' ||
        v_guest.REGISTRATION_DATE
    );
END;


-- Definition of custom record type TYPE data type IS RECORD (…)
CREATE OR REPLACE PROCEDURE guest_info AS
    TYPE t_guest_detail IS RECORD (
        GUEST_ID  GUEST.guest_id%TYPE,
        FIRSTNAME GUEST.FIRSTNAME%TYPE,
        LASTNAME  GUEST.LASTNAME%TYPE,
        CITY      GUEST.CITY%TYPE,
        YEARS_CUSTOMER NUMBER
    );

    v_guest_detail t_guest_detail;
BEGIN
    SELECT GUEST_ID,
           FIRSTNAME,
           LASTNAME,
           CITY,
           ROUND((SYSDATE - REGISTRATION_DATE), 1)
    INTO v_guest_detail
    FROM GUEST
    WHERE GUEST_ID = 1;

    DBMS_OUTPUT.PUT_LINE(
        v_guest_detail.FIRSTNAME || ' ' ||
        v_guest_detail.LASTNAME  || ' lives in: ' ||
        v_guest_detail.CITY      || ' customer for ' ||
        v_guest_detail.YEARS_CUSTOMER || ' days'
    );
END;



-- Tabulka (pole) INDEXED BY TABLE
CREATE OR REPLACE PROCEDURE guest_info_table AS
    TYPE t_guest_table IS TABLE OF GUEST%ROWTYPE INDEXED BY PLS_INTEGER;
    v_guest_table t_guest_table;
    v_counter     PLS_INTEGER := 0;

BEGIN
    SELECT *
    BULK COLLECT INTO v_guest_table
    FROM GUEST;

    v_counter := v_guest_table.COUNT;

    FOR i IN 1..v_counter LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(i).FIRSTNAME || ' ' ||
            v_guest_table(i).LASTNAME  || ' lives in: ' ||
            v_guest_table(i).CITY      || ' and customer registrated from ' ||
            v_guest_table(i).REGISTRATION_DATE
        );
    END LOOP;
END;


--  INDEXED BY TEBLE OF RECORDS
CREATE OR REPLACE PROCEDURE guest_info_table_of_records AS
    TYPE t_guest_record IS RECORD (
        GUEST_ID  GUEST.guest_id%TYPE,
        FIRSTNAME GUEST.FIRSTNAME%TYPE,
        LASTNAME  GUEST.LASTNAME%TYPE,
        CITY      GUEST.CITY%TYPE,
        YEARS_CUSTOMER NUMBER
    );

    TYPE t_guest_table IS TABLE OF t_guest_record INDEXED BY PLS_INTEGER;
    v_guest_table t_guest_table;
    v_counter     PLS_INTEGER := 0;

BEGIN
    SELECT GUEST_ID,
           FIRSTNAME,
           LASTNAME,
           CITY,
           ROUND((SYSDATE - REGISTRATION_DATE), 1)
    BULK COLLECT INTO v_guest_table
    FROM GUEST;

    v_counter := v_guest_table.COUNT;

    FOR i IN 1..v_counter LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(i).FIRSTNAME || ' ' ||
            v_guest_table(i).LASTNAME  || ' lives in: ' ||
            v_guest_table(i).CITY      || ' customer for ' ||
            v_guest_table(i).YEARS_CUSTOMER || ' days'
        );
    END LOOP;
END;


--  INDEX BY BINARY_INTEGER
CREATE OR REPLACE PROCEDURE guest_info_indexed_by_binary_integer AS
    TYPE t_guest_record IS RECORD (
        GUEST_ID  GUEST.guest_id%TYPE,
        FIRSTNAME GUEST.FIRSTNAME%TYPE,
        LASTNAME  GUEST.LASTNAME%TYPE,
        CITY      GUEST.CITY%TYPE,
        YEARS_CUSTOMER NUMBER
    );

    TYPE t_guest_table IS TABLE OF t_guest_record INDEXED BY BINARY_INTEGER;
    v_guest_table t_guest_table;
    v_counter     PLS_INTEGER := 0; -- PLS_INTEGER is preferred for loop counters

BEGIN
    SELECT GUEST_ID,
           FIRSTNAME,
           LASTNAME,
           CITY,
           ROUND((SYSDATE - REGISTRATION_DATE), 1)
    BULK COLLECT INTO v_guest_table
    FROM GUEST;

    v_counter := v_guest_table.COUNT;

    FOR i IN 1..v_counter LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(i).FIRSTNAME || ' ' ||
            v_guest_table(i).LASTNAME  || ' lives in: ' ||
            v_guest_table(i).CITY      || ' customer for ' ||
            v_guest_table(i).YEARS_CUSTOMER || ' days'
        );
    END LOOP;
END;
