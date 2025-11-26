-- SQL script for lesson 5 tasks
-- Richard Ficek 2025
-- Database schema: HOTEL_MANAGEMENT


-- Running script for calling created Cursors procedures
BEGIN
    CHECK_GUEST;
    CHECK_GUEST_RECORD;
    CHECK_GUEST_FOR_LOOP;
    CHECK_GUEST_TYPE('NEW');
    UPDATE_GUEST_TYPE;
    CHECK_GUEST_NESTED;
END;

-- Implement an explicit cursor in the CURSOR declaration in the body of the OPEN-FETCHCLOSE procedure
CREATE OR REPLACE PROCEDURE CHECK_GUEST IS
    CURSOR guest_cursor IS
        SELECT FIRSTNAME, LASTNAME, BIRTH_DATE FROM guest;
    v_first_name guest.LASTNAME%TYPE;
    v_last_name guest.LASTNAME%TYPE;
    v_birth_date guest.BIRTH_DATE%TYPE;
BEGIN
    OPEN guest_cursor;
    LOOP
        FETCH guest_cursor INTO v_first_name, v_last_name, v_birth_date;
        EXIT WHEN guest_cursor%NOTFOUND;
        IF SYSDATE - v_birth_date = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || v_first_name || ' ' || v_last_name || ' has a birthday today!');
        ELSE
            DBMS_OUTPUT.PUT_LINE('Guest ' || v_first_name || ' ' || v_last_name || ' do not has a birthday today!');
        END IF;

    END LOOP;
    CLOSE guest_cursor;
END CHECK_GUEST;

-- Implement a cursor with loading into an INTO record in the %ROWTYPE declaration Use the %ISOPEN, %NOTFOUND, %FOUND, %ROWCOUNT flags in the cursor implementation
CREATE OR REPLACE PROCEDURE CHECK_GUEST_RECORD IS
    CURSOR guest_cursor IS
        SELECT firstname, lastname, birth_date FROM guest;

    guest_record guest_cursor%ROWTYPE;
    v_count NUMBER := 0;
BEGIN
    OPEN guest_cursor;

    LOOP
        FETCH guest_cursor INTO guest_record;
        EXIT WHEN guest_cursor%NOTFOUND;

        -- Might be safer: TRUNC(SYSDATE) = TRUNC(birth_date)
        IF TRUNC(SYSDATE) - TRUNC(guest_record.birth_date) = 0 THEN
            DBMS_OUTPUT.PUT_LINE(
                'Guest ' || guest_record.firstname || ' ' || guest_record.lastname ||
                ' has a birthday today!'
            );
        END IF;
    END LOOP;

    -- Save rowcount BEFORE closing the cursor
    v_count := guest_cursor%ROWCOUNT;

    CLOSE guest_cursor;

    IF NOT guest_cursor%ISOPEN THEN
        DBMS_OUTPUT.PUT_LINE('Cursor is now CLOSED');
    END IF;

    DBMS_OUTPUT.PUT_LINE('Total rows processed: ' || v_count);
END CHECK_GUEST_RECORD;

-- Implement a cursor FOR record name LOOP cursor name END LOOP also use EXIT WHEN inside the cursor
CREATE OR REPLACE PROCEDURE CHECK_GUEST_FOR_LOOP IS
    CURSOR guest_cursor IS
        SELECT FIRSTNAME, LASTNAME, BIRTH_DATE FROM guest;
BEGIN
    FOR guest_record IN guest_cursor LOOP
        IF SYSDATE - guest_record.BIRTH_DATE = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.FIRSTNAME || ' ' || guest_record.LASTNAME || ' has a birthday today!');
        ELSE
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.FIRSTNAME || ' ' || guest_record.LASTNAME || ' do not has a birthday today!');
        END IF;
    END LOOP;
END CHECK_GUEST_FOR_LOOP;

-- Implement a cursor with parameters to be entered in the body of the procedure CURSOR name (parameters) IS …
CREATE OR REPLACE PROCEDURE CHECK_GUEST_TYPE(
    P_GUEST_TYPE IN VARCHAR2
) IS
    CURSOR guest_cursor IS
        SELECT g.FIRSTNAME, g.LASTNAME, g.BIRTH_DATE
        FROM guest g
        JOIN reservation r ON g.GUEST_ID = r.GUEST_ID
        WHERE r.STATUS = P_GUEST_TYPE;
BEGIN
    FOR guest_record IN guest_cursor LOOP
        IF SYSDATE - guest_record.BIRTH_DATE = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.FIRSTNAME || ' ' || guest_record.LASTNAME || ' has a birthday today!');
        END IF;
    END LOOP;
END CHECK_GUEST_TYPE;



-- Implement cursor for UPDATE with WAIT and NOWAIT
CREATE OR REPLACE PROCEDURE UPDATE_GUEST_TYPE IS
    CURSOR guest_type_cursor IS
        SELECT FIRSTNAME, LASTNAME, registration_date, guest_type FROM guest
        -- FOR UPDATE OF first_name NOWAIT;
        FOR UPDATE WAIT 5;
    guest_record guest_type_cursor%ROWTYPE;
BEGIN
    OPEN guest_type_cursor;
    LOOP
        FETCH guest_type_cursor INTO guest_record;
        EXIT WHEN guest_type_cursor%NOTFOUND;

       IF guest_record.REGISTRATION_DATE < ADD_MONTHS(SYSDATE, -6) THEN
            guest_record.GUEST_TYPE := 'REGULAR';
        ELSE
            guest_record.GUEST_TYPE := 'NEW';
        END IF;

        UPDATE guest
        SET guest_type = guest_record.GUEST_TYPE
        WHERE CURRENT OF guest_type_cursor;

        DBMS_OUTPUT.PUT_LINE('Updated guest ' || guest_record.FIRSTNAME || ' ' || guest_record.LASTNAME || ' to type ' || guest_record.GUEST_TYPE);

    END LOOP;
    CLOSE guest_type_cursor;
END UPDATE_GUEST_TYPE;

-- Implement multiple nested cursors, the inner cursor is affected by the outer cursor parameter
CREATE OR REPLACE PROCEDURE CHECK_GUEST_NESTED IS
    CURSOR outer_cursor IS
        SELECT GUEST_ID, FIRSTNAME FROM guest;
    CURSOR inner_cursor (p_guest_id IN NUMBER) IS
        SELECT ROOM.ROOM_ID, room_number FROM reservation
        JOIN room ON reservation.room_id = room.room_id
        WHERE guest_id = p_guest_id;
BEGIN
    FOR outer_record IN outer_cursor LOOP
        DBMS_OUTPUT.PUT_LINE('Processing outer guest: ' || outer_record.FIRSTNAME);
        FOR inner_record IN inner_cursor(outer_record.guest_id) LOOP
            DBMS_OUTPUT.PUT_LINE('  Found room: ' || inner_record.room_number);
        END LOOP;
    END LOOP;
END CHECK_GUEST_NESTED;

