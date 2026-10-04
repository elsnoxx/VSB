

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
    TYPE t_guest_table IS TABLE OF GUEST%ROWTYPE;
    v_guest_table t_guest_table;
BEGIN
    SELECT *
    BULK COLLECT INTO v_guest_table
    FROM GUEST;

    FOR i IN 1..v_guest_table.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(i).firstname || ' ' ||
            v_guest_table(i).lastname  || ' lives in: ' ||
            v_guest_table(i).city      || ' and customer registered from ' ||
            v_guest_table(i).registration_date
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

    TYPE t_guest_table IS TABLE OF t_guest_record INDEX BY PLS_INTEGER;
    v_guest_table t_guest_table;

    i PLS_INTEGER := 1;
BEGIN
    FOR rec IN (
        SELECT GUEST_ID,
               FIRSTNAME,
               LASTNAME,
               CITY,
               ROUND((SYSDATE - REGISTRATION_DATE), 1) AS YEARS_CUSTOMER
        FROM GUEST
    ) LOOP
        v_guest_table(i).GUEST_ID       := rec.GUEST_ID;
        v_guest_table(i).FIRSTNAME      := rec.FIRSTNAME;
        v_guest_table(i).LASTNAME       := rec.LASTNAME;
        v_guest_table(i).CITY           := rec.CITY;
        v_guest_table(i).YEARS_CUSTOMER := rec.YEARS_CUSTOMER;
        i := i + 1;
    END LOOP;

    FOR j IN 1 .. v_guest_table.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(j).FIRSTNAME || ' ' ||
            v_guest_table(j).LASTNAME  || ' lives in: ' ||
            v_guest_table(j).CITY      || ', customer for ' ||
            v_guest_table(j).YEARS_CUSTOMER || ' days'
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

    TYPE t_guest_table IS TABLE OF t_guest_record INDEX BY BINARY_INTEGER;
    v_guest_table t_guest_table;

    i BINARY_INTEGER := 1;
BEGIN
    FOR rec IN (
        SELECT GUEST_ID,
               FIRSTNAME,
               LASTNAME,
               CITY,
               ROUND((SYSDATE - REGISTRATION_DATE), 1) AS YEARS_CUSTOMER
        FROM GUEST
    ) LOOP
        v_guest_table(i).GUEST_ID       := rec.GUEST_ID;
        v_guest_table(i).FIRSTNAME      := rec.FIRSTNAME;
        v_guest_table(i).LASTNAME       := rec.LASTNAME;
        v_guest_table(i).CITY           := rec.CITY;
        v_guest_table(i).YEARS_CUSTOMER := rec.YEARS_CUSTOMER;
        i := i + 1;
    END LOOP;

    FOR j IN 1 .. v_guest_table.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(
            v_guest_table(j).FIRSTNAME || ' ' ||
            v_guest_table(j).LASTNAME  || ' lives in: ' ||
            v_guest_table(j).CITY      || ', customer for ' ||
            v_guest_table(j).YEARS_CUSTOMER || ' days'
        );
    END LOOP;
END;
