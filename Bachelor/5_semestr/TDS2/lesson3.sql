-- • Lesson 3
--      o Write a procedure that will contain INSERT, UPDATE, DELETE, MERGE
--      o Use in procedure SELECT atr INTO prom …
--      o Use SQL%FOUND, SQL%NOTFOUND, SQL%ROWCOUNT as part of data manipulation
--      o Use COMMIT, ROLLBACK, SAVEPOINT within the transaction handling procedure
--

CREATE OR REPLACE PROCEDURE manage_service_price_history(
    p_sph_id IN NUMBER,
    new_price IN SERVICEPRICEHISTORY.price%Type,
    p_valid_to IN SERVICEPRICEHISTORY.VALID_TO%Type,
    p_service_id NUMBER,
    dell_flag IN BOOLEAN,
    new_flag IN BOOLEAN,
    merge_flag IN BOOLEAN DEFAULT FALSE  -- Přidáno pro MERGE
) IS
    old_price SERVICEPRICEHISTORY.price%TYPE;
    v_count NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('Start manage_service_price_history for SPH_ID=' || p_sph_id);

    -- SELECT INTO (máte)
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

    -- SAVEPOINT (máte)
    SAVEPOINT before_change;

    -- DELETE (máte) + SQL%ROWCOUNT (máte)
    IF dell_flag THEN
        DELETE FROM SERVICEPRICEHISTORY WHERE SPH_ID = p_sph_id;
        IF SQL%ROWCOUNT > 0 THEN
            DBMS_OUTPUT.PUT_LINE('Service price history deleted successfully. Rows affected: ' || SQL%ROWCOUNT);
        ELSE
            DBMS_OUTPUT.PUT_LINE('No service price history to delete, rolling back.');
            ROLLBACK TO before_change;
        END IF;
    END IF;

    -- MERGE (CHYBĚLO VÁM!)
    IF merge_flag THEN
        MERGE INTO SERVICEPRICEHISTORY sph
        USING (SELECT new_price as price, p_valid_to as valid_to, p_service_id as service_id FROM dual) src
        ON (sph.SPH_ID = p_sph_id)
        WHEN MATCHED THEN
            UPDATE SET PRICE = src.price, VALID_TO = src.valid_to
        WHEN NOT MATCHED THEN
            INSERT (PRICE, VALID_FROM, VALID_TO, SERVICE_ID)
            VALUES (src.price, SYSDATE, src.valid_to, src.service_id);
        
        -- SQL%FOUND (CHYBĚLO VÁM!)
        IF SQL%FOUND THEN
            DBMS_OUTPUT.PUT_LINE('MERGE operation successful. Rows processed: ' || SQL%ROWCOUNT);
        END IF;
        
        -- SQL%NOTFOUND (CHYBĚLO VÁM!)
        IF SQL%NOTFOUND THEN
            DBMS_OUTPUT.PUT_LINE('MERGE operation found no matching records.');
        END IF;
    END IF;

    -- INSERT (máte)
    IF new_flag AND NOT merge_flag THEN
        INSERT INTO SERVICEPRICEHISTORY (PRICE, VALID_FROM, VALID_TO, SERVICE_ID)
        VALUES (new_price, SYSDATE, p_valid_to, p_service_id);
        
        -- Přidáno SQL%FOUND/NOTFOUND
        IF SQL%FOUND THEN
            DBMS_OUTPUT.PUT_LINE('INSERT successful. Rows inserted: ' || SQL%ROWCOUNT);
        END IF;
    END IF;

    -- UPDATE (máte)
    IF NOT new_flag AND NOT merge_flag THEN
        UPDATE SERVICEPRICEHISTORY
        SET PRICE = new_price, VALID_TO = p_valid_to
        WHERE SPH_ID = p_sph_id;
        
        -- Přidáno SQL%FOUND/NOTFOUND/ROWCOUNT
        IF SQL%FOUND THEN
            DBMS_OUTPUT.PUT_LINE('UPDATE successful. Rows updated: ' || SQL%ROWCOUNT);
        ELSIF SQL%NOTFOUND THEN
            DBMS_OUTPUT.PUT_LINE('UPDATE found no matching records to update.');
            ROLLBACK TO before_change;
        END IF;
    END IF;

    DBMS_OUTPUT.PUT_LINE('Finished manage_service_price_history for SPH_ID=' || p_sph_id);
    COMMIT; -- COMMIT (máte)

EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Unexpected error: ' || SQLERRM);
        ROLLBACK TO before_change; -- ROLLBACK (máte)
END;