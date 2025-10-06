-- SQL script for lesson 4 tasks
-- Richard Ficek 2025
-- Database schema: HOTEL_MANAGEMENT


-- Running script for calling created procedures
BEGIN
    -- Call procedures here for testing
    FEEDBACK_RATING(1);
    GUEST_TYPE(1);
    ROOM_TYPE(101);
    CHECK_LOGICAL_OPERATORS('TRUE', 'FALSE');
    SIMPLE_COUNTDOWN(5);
    WHILE_COUNTDOWN(5);
    FOR_COUNTDOWN(5);
    REVERSE_COUNTDOWN(5);
    NESTED_LOOPS_EXIT_LABELS;
END;



-- Implement a procedure with IF-THEN-ELSE and IF-ELSEIF-ELSE

CREATE OR REPLACE PROCEDURE FEEDBACK_RATING(
    P_FEEDBACK_ID IN NUMBER,
    
) IS
    v_result NUMBER;
BEGIN
    SELECT RATING(P_FEEDBACK_ID) INTO v_result FROM FEEDBACK WHERE FEEDBACK_ID = P_FEEDBACK_ID;

    IF  v_result = 5 THEN
        DBMS_OUTPUT.PUT_LINE('Satisfied');
    ELSIF v_result = 4 THEN
        DBMS_OUTPUT.PUT_LINE('Neutral');
    ELSIF v_result = 3 THEN
        DBMS_OUTPUT.PUT_LINE('Dissatisfied');
    ELSIF v_result = 2 THEN
        DBMS_OUTPUT.PUT_LINE('Very Dissatisfied');
    ELSIF v_result = 1 THEN
        DBMS_OUTPUT.PUT_LINE('Extremely Dissatisfied');
    ELSE
        DBMS_OUTPUT.PUT_LINE('No rating available');
    END IF;


EXCEPTION
    WHEN NO_DATA_FOUND THEN
        v_result := NULL;
        DBMS_OUTPUT.PUT_LINE('No feedback found with the given ID.');
    WHEN OTHERS THEN
        RAISE;
    
    COMMIT;
END;


--  Implement a procedure with CASE-WHEN-ELSE (with a variable after the CASE) as a separate implementation, or as the result of an assignment to a variable
CREATE OR REPLACE PROCEDURE GUEST_TYPE(
    P_GUEST_ID IN NUMBER
) IS
    v_result NUMBER;
    v_guest_type VARCHAR2(20);
BEGIN
    SELECT COUNT(*) INTO v_result FROM RESERVATION WHERE GUEST_ID = P_GUEST_ID;

    CASE
        WHEN v_result BETWEEN 1 AND 5 THEN
            v_guest_type := 'New Guest';
        WHEN v_result BETWEEN 6 AND 15 THEN
            v_guest_type := 'Regular Guest';
        WHEN v_result > 15 THEN
            v_guest_type := 'Loyal Guest';
        ELSE
            v_guest_type := 'New Guest';
    END CASE;

    DBMS_OUTPUT.PUT_LINE('Guest Type: ' || v_guest_type);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        v_result := NULL;
        DBMS_OUTPUT.PUT_LINE('No reservations found for the given guest ID.');
    WHEN OTHERS THEN
        RAISE;

END;


-- Implement CASE-WHEN-ELSE with a condition after WHEN
CREATE OR REPLACE PROCEDURE ROOM_TYPE(
    P_ROOM_ID IN NUMBER
) IS
    v_room_type VARCHAR2(20);
BEGIN
    SELECT TYPE INTO v_room_type FROM ROOM WHERE ROOM_ID = P_ROOM_ID;

    CASE v_room_type
        WHEN 'SINGLE' THEN
            DBMS_OUTPUT.PUT_LINE('Single Room');
        WHEN 'DOUBLE' THEN
            DBMS_OUTPUT.PUT_LINE('Double Room');
        WHEN 'SUITE' THEN
            DBMS_OUTPUT.PUT_LINE('Suite Room');
        ELSE
            DBMS_OUTPUT.PUT_LINE('Unknown Room Type');
    END CASE;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No room found with the given ID.');
    WHEN OTHERS THEN
        RAISE;
END;



-- Implement a procedure solving all variants of combinations of logical operators AND, OR, NOT with input values TRUE, FALSE, NULL
CREATE OR REPLACE PROCEDURE CHECK_LOGICAL_OPERATORS(
    P_VAL1 IN VARCHAR2,
    P_VAL2 IN VARCHAR2
) IS
    v_result VARCHAR2(50);
BEGIN
    v_result := (P_VAL1 AND P_VAL2) OR (P_VAL1 NOT P_VAL2);
    IF v_result IS NULL THEN
        DBMS_OUTPUT.PUT_LINE('Result is NULL');
    ELSIF v_result = 'TRUE' THEN
        DBMS_OUTPUT.PUT_LINE('Result is TRUE');
    ELSIF v_result = 'FALSE' THEN
        DBMS_OUTPUT.PUT_LINE('Result is FALSE');
    ELSE
        DBMS_OUTPUT.PUT_LINE('Unexpected result');
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE;
END; 

--  Implement a loop with LOOP – EXIT [WHEN] – END LOOP
CREATE OR REPLACE PROCEDURE SIMPLE_COUNTDOWN(
    P_START IN NUMBER
) IS
    v_counter NUMBER := P_START;
BEGIN
    LOOP
        DBMS_OUTPUT.PUT_LINE('Countdown: ' || v_counter);
        v_counter := v_counter - 1;
        EXIT WHEN v_counter < 1;
    END LOOP;
END;

--  Implement a WHILE loop - LOOP condition - END LOOP
CREATE OR REPLACE PROCEDURE WHILE_COUNTDOWN(
    P_START IN NUMBER
) IS
    v_counter NUMBER := P_START;
BEGIN
    WHILE v_counter >= 1 LOOP
        DBMS_OUTPUT.PUT_LINE('Countdown: ' || v_counter);
        v_counter := v_counter - 1;
    END LOOP;
END;

--  Implement the cycle FOR counter IN lower..upper LOOP – END LOOP
CREATE OR REPLACE PROCEDURE FOR_COUNTDOWN(
    P_START IN NUMBER
) IS
BEGIN
    FOR v_counter IN REVERSE 1..P_START LOOP
        DBMS_OUTPUT.PUT_LINE('Countdown: ' || v_counter);
    END LOOP;
END;

--   Implement a loop with a reverse counter REVERSE
CREATE OR REPLACE PROCEDURE REVERSE_COUNTDOWN(
    P_START IN NUMBER
) IS
BEGIN
    FOR v_counter IN REVERSE 1..P_START LOOP
        DBMS_OUTPUT.PUT_LINE('Countdown: ' || v_counter);
    END LOOP;
END;

--  Implement nested loops with EXIT termination and labels
CREATE OR REPLACE PROCEDURE NESTED_LOOPS_EXIT_LABELS IS
    v_outer_counter NUMBER := 1;
    v_inner_counter NUMBER := 1;
BEGIN
    <<outer_loop>>
    LOOP
        DBMS_OUTPUT.PUT_LINE('Outer Loop Counter: ' || v_outer_counter);
        v_inner_counter := 1;
        
        <<inner_loop>>
        LOOP
            DBMS_OUTPUT.PUT_LINE('  Inner Loop Counter: ' || v_inner_counter);
            v_inner_counter := v_inner_counter + 1;
            IF v_inner_counter > 3 THEN
                EXIT inner_loop; -- Exit inner loop when counter exceeds 3
            END IF;
        END LOOP inner_loop;
        
        v_outer_counter := v_outer_counter + 1;
        IF v_outer_counter > 2 THEN
            EXIT outer_loop; -- Exit outer loop when counter exceeds 2
        END IF;
    END LOOP outer_loop;
END;