

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

