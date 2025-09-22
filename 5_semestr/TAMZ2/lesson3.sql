-- • Lesson 3
--      o Write a procedure that will contain INSERT, UPDATE, DELETE, MERGE
--      o Use in procedure SELECT atr INTO prom …
--      o Use SQL%FOUND, SQL%NOTFOUND, SQL%ROWCOUNT as part of data manipulation
--      o Use COMMIT, ROLLBACK, SAVEPOINT within the transaction handling procedure

CREATE OR REPLACE PROCEDURE manage_service_price_history(
    p_sph_id IN NUMBER,
    new_price IN SERVICEPRICEHISTORY.price%Type,
    p_valid_to IN SERVICEPRICEHISTORY.VALID_TO%Type,
    p_service_id NUMBER,
    dell_flag IN BOOLEAN,
    new_flag IN BOOLEAN
) IS
    old_price SERVICEPRICEHISTORY.price%TYPE;
BEGIN
    DBMS_OUTPUT.PUT_LINE('Start manage_service_price_history for SPH_ID=' || p_sph_id);

    -- načtení staré ceny
    BEGIN
        SELECT PRICE
        INTO old_price
        FROM SERVICEPRICEHISTORY
        WHERE SPH_ID = p_sph_id;
        DBMS_OUTPUT.PUT_LINE('Old price: ' || old_price);
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            DBMS_OUTPUT.PUT_LINE('No existing price found for SPH_ID=' || p_sph_id);
            old_price := NULL;
    END;

    -- SAVEPOINT
    SAVEPOINT before_change;

    -- Delete
    IF dell_flag THEN
        DELETE FROM SERVICEPRICEHISTORY WHERE SPH_ID = p_sph_id;
        IF SQL%ROWCOUNT > 0 THEN
            DBMS_OUTPUT.PUT_LINE('Service price history deleted successfully.');
        ELSE
            DBMS_OUTPUT.PUT_LINE('No service price history to delete, rolling back.');
            ROLLBACK TO before_change;
        END IF;
    END IF;

    -- Insert or Update
    IF new_flag THEN
        INSERT INTO SERVICEPRICEHISTORY (PRICE, VALID_FROM, VALID_TO, SERVICE_ID)
        VALUES ( new_price, SYSDATE, p_valid_to, p_service_id);
        DBMS_OUTPUT.PUT_LINE('Inserted new record: price=' || new_price || ', valid_to=' || p_valid_to);
    ELSE
        -- zkontroluj, zda řádek existuje
        DECLARE
            v_count NUMBER;
        BEGIN
            SELECT COUNT(*) INTO v_count
            FROM SERVICEPRICEHISTORY
            WHERE SPH_ID = p_sph_id;

            IF v_count > 0 THEN
                UPDATE SERVICEPRICEHISTORY
                SET PRICE = new_price,
                    VALID_TO = p_valid_to
                WHERE SPH_ID = p_sph_id;
                DBMS_OUTPUT.PUT_LINE('Updated record to new price=' || new_price || ', valid_to=' || p_valid_to);
            ELSE
                DBMS_OUTPUT.PUT_LINE('Service price history not found, rolling back.');
                ROLLBACK TO before_change;
            END IF;
        END;
    END IF;

    DBMS_OUTPUT.PUT_LINE('Finished manage_service_price_history for SPH_ID=' || p_sph_id);
    COMMIT;

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Unexpected error: ' || SQLERRM);
        ROLLBACK TO before_change;
END;