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
        SELECT first_name, last_name, birth_date FROM guest;
    v_first_name guest.first_name%TYPE;
    v_last_name guest.last_name%TYPE;
    v_birth_date guest.birth_date%TYPE;
BEGIN
    OPEN guest_cursor;
    LOOP
        FETCH guest_cursor INTO v_first_name, v_last_name, v_birth_date;
        EXIT WHEN guest_cursor%NOTFOUND;
        IF SYSDATE - v_birth_date = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || v_first_name || ' ' || v_last_name || ' has a birthday today!');
        END IF;

    END LOOP;
    CLOSE guest_cursor;
END CHECK_GUEST;

-- Implement a cursor with loading into an INTO record in the %ROWTYPE declaration Use the %ISOPEN, %NOTFOUND, %FOUND, %ROWCOUNT flags in the cursor implementation
CREATE OR REPLACE PROCEDURE CHECK_GUEST_RECORD IS
    CURSOR guest_cursor IS
        SELECT first_name, last_name, birth_date FROM guest;
    guest_record guest_cursor%ROWTYPE;
BEGIN
    OPEN guest_cursor;
    LOOP
        FETCH guest_cursor INTO guest_record;
        EXIT WHEN guest_cursor%NOTFOUND;
        IF SYSDATE - guest_record.birth_date = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.first_name || ' ' || guest_record.last_name || ' has a birthday today!');
        END IF;

    END LOOP;
    
    CLOSE guest_cursor;

    IF NOT guest_cursor%ISOPEN THEN
        DBMS_OUTPUT.PUT_LINE('Cursor is now CLOSED');
    END IF;
    DBMS_OUTPUT.PUT_LINE('Total rows processed: ' || guest_cursor%ROWCOUNT);
END CHECK_GUEST_RECORD;

-- Implement a cursor FOR record name LOOP cursor name END LOOP also use EXIT WHEN inside the cursor
CREATE OR REPLACE PROCEDURE CHECK_GUEST_FOR_LOOP IS
    CURSOR guest_cursor IS
        SELECT first_name, last_name, birth_date FROM guest;
BEGIN
    FOR guest_record IN guest_cursor LOOP
        IF SYSDATE - guest_record.birth_date = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.first_name || ' ' || guest_record.last_name || ' has a birthday today!');
        END IF;
    END LOOP;
END CHECK_GUEST_FOR_LOOP;

-- Implement a cursor with parameters to be entered in the body of the procedure CURSOR name (parameters) IS …
CREATE OR REPLACE PROCEDURE CHECK_GUEST_TYPE(
    P_GUEST_TYPE IN VARCHAR2
) IS
    CURSOR guest_cursor IS
        SELECT g.first_name, g.last_name, g.birth_date
        FROM guest g
        JOIN reservation r ON g.guest_id = r.guest_id
        WHERE r.status = P_GUEST_TYPE;
BEGIN
    FOR guest_record IN guest_cursor LOOP
        IF SYSDATE - guest_record.birth_date = 0 THEN
            DBMS_OUTPUT.PUT_LINE('Guest ' || guest_record.first_name || ' ' || guest_record.last_name || ' has a birthday today!');
        END IF;
    END LOOP;
END CHECK_GUEST_TYPE;


-- Implement cursor for UPDATE with WAIT and NOWAIT
CREATE OR REPLACE PROCEDURE UPDATE_GUEST_TYPE IS
    CURSOR guest_type_cursor IS
        SELECT first_name, last_name, registration_date, guest_type FROM guest
        -- FOR UPDATE OF first_name NOWAIT;
        FOR UPDATE WAIT 5;
    guest_record guest_type_cursor%ROWTYPE;
BEGIN
    OPEN guest_type_cursor;
    LOOP
        FETCH guest_type_cursor INTO guest_record;
        EXIT WHEN guest_type_cursor%NOTFOUND;

       IF guest_record.registration_date < ADD_MONTHS(SYSDATE, -6) THEN
            guest_record.guest_type := 'REGULAR';
        ELSE
            guest_record.guest_type := 'NEW';
        END IF;

        UPDATE guest
        SET guest_type = guest_record.guest_type
        WHERE CURRENT OF guest_type_cursor;
        
        DBMS_OUTPUT.PUT_LINE('Updated guest ' || guest_record.first_name || ' ' || guest_record.last_name || ' to type ' || guest_record.guest_type);

    END LOOP;
    CLOSE guest_type_cursor;
END UPDATE_GUEST_TYPE;

-- Implement multiple nested cursors, the inner cursor is affected by the outer cursor parameter
CREATE OR REPLACE PROCEDURE CHECK_GUEST_NESTED IS
    CURSOR outer_cursor IS
        SELECT guest_id, first_name FROM guest;
    CURSOR inner_cursor (p_guest_id IN NUMBER) IS
        SELECT room_id, room_number FROM reservation 
        JOIN room ON reservation.room_id = room.room_id
        WHERE guest_id = p_guest_id;
BEGIN
    FOR outer_record IN outer_cursor LOOP
        DBMS_OUTPUT.PUT_LINE('Processing outer guest: ' || outer_record.first_name);
        FOR inner_record IN inner_cursor(outer_record.guest_id) LOOP
            DBMS_OUTPUT.PUT_LINE('  Found room: ' || inner_record.room_number);
        END LOOP;
    END LOOP;
END CHECK_GUEST_NESTED;
