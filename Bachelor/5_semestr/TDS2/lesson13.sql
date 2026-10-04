/*
===============================================================================
LEKCE 13: TRIGGERY V ORACLE DATABASE
===============================================================================
Obsah:
1. Triggery BEFORE/AFTER FOR EACH ROW
2. INSTEAD OF trigger pro složený pohled
3. Triggery s :NEW a :OLD proměnnými
4. Větvené triggery pro více operací (INSERT/UPDATE/DELETE)
5. Demonstrace mutujících tabulek (Mutation Error)
===============================================================================
*/

-- Čištění před začátkem
SET SERVEROUTPUT ON SIZE 1000000;

-- =============================================
-- 1. PŘÍPRAVNÉ TABULKY PRO AUDIT A LOG
-- =============================================

-- Audit tabulka pro sledování změn
CREATE TABLE audit_log (
    log_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    table_name VARCHAR2(50),
    operation VARCHAR2(10),
    user_name VARCHAR2(50),
    timestamp DATE DEFAULT SYSDATE,
    old_values CLOB,
    new_values CLOB
);

-- Statistická tabulka pro počítání operací
CREATE TABLE operation_stats (
    table_name VARCHAR2(50) PRIMARY KEY,
    insert_count NUMBER DEFAULT 0,
    update_count NUMBER DEFAULT 0,
    delete_count NUMBER DEFAULT 0,
    last_operation DATE
);

-- Inicializace statistik pro hlavní tabulky
INSERT ALL
    INTO operation_stats (table_name) VALUES ('Guest')
    INTO operation_stats (table_name) VALUES ('Reservation')
    INTO operation_stats (table_name) VALUES ('Room')
    INTO operation_stats (table_name) VALUES ('Payment')
SELECT 1 FROM dual;
COMMIT;

-- =============================================
-- 2. BEFORE/AFTER TRIGGERY FOR EACH ROW
-- =============================================

-- A) BEFORE INSERT trigger - validace a úprava dat před vložením
CREATE OR REPLACE TRIGGER trg_guest_before_insert
    BEFORE INSERT ON Guest
    FOR EACH ROW
BEGIN
    -- Validace věku (musí být starší 18 let)
    IF :NEW.birth_date > SYSDATE - (18 * 365) THEN
        RAISE_APPLICATION_ERROR(-20001, 'Host musí být starší 18 let');
    END IF;
    
    -- Automatické nastavení registračního data
    IF :NEW.registration_date IS NULL THEN
        :NEW.registration_date := SYSDATE;
    END IF;
    
    -- Normalizace emailu na malá písmena
    :NEW.email := LOWER(:NEW.email);
    
    -- Výchozí typ hosta
    IF :NEW.guest_type IS NULL THEN
        :NEW.guest_type := 'Regular';
    END IF;
    
    DBMS_OUTPUT.PUT_LINE('BEFORE INSERT: Validace a úprava dat pro nového hosta dokončena');
END;
/

-- B) AFTER INSERT trigger - audit a statistiky
CREATE OR REPLACE TRIGGER trg_guest_after_insert
    AFTER INSERT ON Guest
    FOR EACH ROW
BEGIN
    -- Zápis do audit logu
    INSERT INTO audit_log (table_name, operation, user_name, new_values)
    VALUES ('Guest', 'INSERT', USER, 
            'ID: ' || :NEW.guest_id || ', Name: ' || :NEW.firstname || ' ' || :NEW.lastname || 
            ', Email: ' || :NEW.email || ', Type: ' || :NEW.guest_type);
    
    -- Aktualizace statistik
    UPDATE operation_stats 
    SET insert_count = insert_count + 1, 
        last_operation = SYSDATE 
    WHERE table_name = 'Guest';
    
    DBMS_OUTPUT.PUT_LINE('AFTER INSERT: Audit a statistiky pro hosta ID ' || :NEW.guest_id || ' dokončeny');
END;
/

-- C) BEFORE UPDATE trigger - validace změn
CREATE OR REPLACE TRIGGER trg_guest_before_update
    BEFORE UPDATE ON Guest
    FOR EACH ROW
BEGIN
    -- Zabránění změně ID (i když je to identity column)
    IF :OLD.guest_id != :NEW.guest_id THEN
        RAISE_APPLICATION_ERROR(-20002, 'ID hosta nelze měnit');
    END IF;
    
    -- Validace změny emailu
    IF :OLD.email != :NEW.email THEN
        -- Kontrola formátu emailu
        IF NOT REGEXP_LIKE(:NEW.email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$') THEN
            RAISE_APPLICATION_ERROR(-20003, 'Neplatný formát emailu');
        END IF;
        :NEW.email := LOWER(:NEW.email);
    END IF;
    
    DBMS_OUTPUT.PUT_LINE('BEFORE UPDATE: Validace změn pro hosta ID ' || :OLD.guest_id);
END;
/

-- D) AFTER UPDATE trigger
CREATE OR REPLACE TRIGGER trg_guest_after_update
    AFTER UPDATE ON Guest
    FOR EACH ROW
BEGIN
    -- Audit změn
    INSERT INTO audit_log (table_name, operation, user_name, old_values, new_values)
    VALUES ('Guest', 'UPDATE', USER,
            'Email: ' || :OLD.email || ', Type: ' || :OLD.guest_type,
            'Email: ' || :NEW.email || ', Type: ' || :NEW.guest_type);
    
    -- Statistiky
    UPDATE operation_stats 
    SET update_count = update_count + 1, 
        last_operation = SYSDATE 
    WHERE table_name = 'Guest';
    
    DBMS_OUTPUT.PUT_LINE('AFTER UPDATE: Host ID ' || :NEW.guest_id || ' aktualizován');
END;
/

-- E) BEFORE DELETE trigger
CREATE OR REPLACE TRIGGER trg_guest_before_delete
    BEFORE DELETE ON Guest
    FOR EACH ROW
DECLARE
    v_reservation_count NUMBER;
BEGIN
    -- Kontrola, zda host nemá aktivní rezervace
    SELECT COUNT(*) INTO v_reservation_count
    FROM Reservation 
    WHERE guest_id = :OLD.guest_id AND status IN ('Confirmed', 'Checked-in');
    
    IF v_reservation_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20004, 'Nelze smazat hosta s aktivními rezervacemi');
    END IF;
    
    DBMS_OUTPUT.PUT_LINE('BEFORE DELETE: Kontrola pro mazání hosta ID ' || :OLD.guest_id);
END;
/

-- =============================================
-- 3. SLOŽENÝ POHLED A INSTEAD OF TRIGGER
-- =============================================

-- Vytvoření složeného pohledu spojujícího více tabulek
CREATE OR REPLACE VIEW v_reservation_details AS
SELECT 
    r.reservation_id,
    r.check_in_date,
    r.check_out_date,
    r.status,
    r.accommodation_price,
    g.guest_id,
    g.firstname,
    g.lastname,
    g.email,
    g.guest_type,
    rm.room_id,
    rm.room_number,
    rt.name AS room_type_name,
    rt.bed_count,
    e.employee_id,
    e.firstname AS employee_firstname,
    e.lastname AS employee_lastname,
    p.payment_id,
    p.total_accommodation,
    p.is_paid
FROM Reservation r
JOIN Guest g ON r.guest_id = g.guest_id
JOIN Room rm ON r.room_id = rm.room_id
JOIN RoomType rt ON rm.room_type_id = rt.room_type_id
JOIN Employee e ON r.employee_id = e.employee_id
JOIN Payment p ON r.payment_id = p.payment_id;

-- INSTEAD OF INSERT trigger pro složený pohled
CREATE OR REPLACE TRIGGER trg_reservation_view_insert
    INSTEAD OF INSERT ON v_reservation_details
    FOR EACH ROW
DECLARE
    v_guest_id NUMBER;
    v_room_id NUMBER;
    v_employee_id NUMBER;
    v_payment_id NUMBER;
    v_room_type_id NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('INSTEAD OF INSERT: Zpracování vložení do pohledu');
    
    -- 1. Vložení nového hosta (pokud neexistuje)
    BEGIN
        SELECT guest_id INTO v_guest_id 
        FROM Guest 
        WHERE email = :NEW.email;
        
        DBMS_OUTPUT.PUT_LINE('Host s emailem ' || :NEW.email || ' již existuje (ID: ' || v_guest_id || ')');
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            INSERT INTO Guest (firstname, lastname, email, birth_date, street, city, postal_code, country, guest_type)
            VALUES (:NEW.firstname, :NEW.lastname, :NEW.email, SYSDATE - (25*365), 
                    'Neznámá adresa', 'Neznámo', '00000', 'Czech Republic', :NEW.guest_type)
            RETURNING guest_id INTO v_guest_id;
            
            DBMS_OUTPUT.PUT_LINE('Vytvořen nový host s ID: ' || v_guest_id);
    END;
    
    -- 2. Nalezení pokoje podle čísla
    BEGIN
        SELECT room_id INTO v_room_id 
        FROM Room 
        WHERE room_number = :NEW.room_number;
    EXCEPTION
        WHEN NO_DATA_FOUND THEN
            -- Vytvoření nového typu pokoje
            INSERT INTO RoomType (name, bed_count) 
            VALUES (:NEW.room_type_name, :NEW.bed_count)
            RETURNING room_type_id INTO v_room_type_id;
            
            -- Vytvoření nového pokoje
            INSERT INTO Room (room_type_id, room_number, is_occupied) 
            VALUES (v_room_type_id, :NEW.room_number, 0)
            RETURNING room_id INTO v_room_id;
            
            DBMS_OUTPUT.PUT_LINE('Vytvořen nový pokoj s ID: ' || v_room_id);
    END;
    
    -- 3. Nalezení zaměstnance
    SELECT employee_id INTO v_employee_id 
    FROM Employee 
    WHERE ROWNUM = 1; -- První dostupný zaměstnanec
    
    -- 4. Vytvoření platby
    INSERT INTO Payment (total_accommodation, total_expenses, is_paid)
    VALUES (:NEW.accommodation_price, 0, NVL(:NEW.is_paid, 0))
    RETURNING payment_id INTO v_payment_id;
    
    -- 5. Vytvoření rezervace
    INSERT INTO Reservation (guest_id, room_id, employee_id, check_in_date, check_out_date, payment_id, status, accommodation_price)
    VALUES (v_guest_id, v_room_id, v_employee_id, :NEW.check_in_date, :NEW.check_out_date, 
            v_payment_id, NVL(:NEW.status, 'Confirmed'), :NEW.accommodation_price);
    
    DBMS_OUTPUT.PUT_LINE('INSTEAD OF INSERT: Rezervace úspěšně vytvořena');
END;
/

-- =============================================
-- 4. TRIGGER S VĚTVENÍM PRO VÍCE OPERACÍ
-- =============================================

CREATE OR REPLACE TRIGGER trg_reservation_all_operations
    BEFORE INSERT OR UPDATE OR DELETE ON Reservation
    FOR EACH ROW
DECLARE
    v_room_status NUMBER;
    v_operation VARCHAR2(10);
BEGIN
    -- Určení typu operace
    IF INSERTING THEN
        v_operation := 'INSERT';
        DBMS_OUTPUT.PUT_LINE('MULTI-TRIGGER: Zpracování INSERT operace');
        
        -- Validace dat pro INSERT
        IF :NEW.check_in_date >= :NEW.check_out_date THEN
            RAISE_APPLICATION_ERROR(-20005, 'Datum check-in musí být před check-out');
        END IF;
        
        -- Kontrola dostupnosti pokoje
        SELECT COUNT(*) INTO v_room_status
        FROM Reservation
        WHERE room_id = :NEW.room_id 
        AND status IN ('Confirmed', 'Checked-in')
        AND ((:NEW.check_in_date BETWEEN check_in_date AND check_out_date)
             OR (:NEW.check_out_date BETWEEN check_in_date AND check_out_date));
        
        IF v_room_status > 0 THEN
            RAISE_APPLICATION_ERROR(-20006, 'Pokoj není v daném termínu dostupný');
        END IF;
        
        -- Nastavení výchozího stavu
        IF :NEW.status IS NULL THEN
            :NEW.status := 'Confirmed';
        END IF;
        
    ELSIF UPDATING THEN
        v_operation := 'UPDATE';
        DBMS_OUTPUT.PUT_LINE('MULTI-TRIGGER: Zpracování UPDATE operace');
        
        -- Zabránění změně ID
        IF :OLD.reservation_id != :NEW.reservation_id THEN
            RAISE_APPLICATION_ERROR(-20007, 'ID rezervace nelze měnit');
        END IF;
        
        -- Kontrola změny stavu
        IF :OLD.status = 'Checked-out' AND :NEW.status != 'Checked-out' THEN
            RAISE_APPLICATION_ERROR(-20008, 'Nelze změnit stav dokončené rezervace');
        END IF;
        
        -- Aktualizace obsazenosti pokoje
        IF :OLD.status != :NEW.status THEN
            IF :NEW.status = 'Checked-in' THEN
                UPDATE Room SET is_occupied = 1 WHERE room_id = :NEW.room_id;
            ELSIF :NEW.status = 'Checked-out' THEN
                UPDATE Room SET is_occupied = 0 WHERE room_id = :NEW.room_id;
            END IF;
        END IF;
        
    ELSIF DELETING THEN
        v_operation := 'DELETE';
        DBMS_OUTPUT.PUT_LINE('MULTI-TRIGGER: Zpracování DELETE operace');
        
        -- Kontrola, zda lze rezervaci smazat
        IF :OLD.status = 'Checked-in' THEN
            RAISE_APPLICATION_ERROR(-20009, 'Nelze smazat aktivní rezervaci');
        END IF;
        
        -- Uvolnění pokoje
        UPDATE Room SET is_occupied = 0 WHERE room_id = :OLD.room_id;
        
        -- Záznam o smazání do audit logu
        INSERT INTO audit_log (table_name, operation, user_name, old_values)
        VALUES ('Reservation', 'DELETE', USER,
                'ID: ' || :OLD.reservation_id || ', Guest: ' || :OLD.guest_id || 
                ', Room: ' || :OLD.room_id || ', Status: ' || :OLD.status);
    END IF;
    
    -- Aktualizace statistik pro všechny operace
    IF INSERTING THEN
        UPDATE operation_stats 
        SET insert_count = insert_count + 1, last_operation = SYSDATE 
        WHERE table_name = 'Reservation';
    ELSIF UPDATING THEN
        UPDATE operation_stats 
        SET update_count = update_count + 1, last_operation = SYSDATE 
        WHERE table_name = 'Reservation';
    ELSIF DELETING THEN
        UPDATE operation_stats 
        SET delete_count = delete_count + 1, last_operation = SYSDATE 
        WHERE table_name = 'Reservation';
    END IF;
    
    DBMS_OUTPUT.PUT_LINE('MULTI-TRIGGER: ' || v_operation || ' operace dokončena');
END;
/

-- =============================================
-- 5. TRIGGER ZPŮSOBUJÍCÍ MUTATION ERROR
-- =============================================

-- Tento trigger způsobí chybu mutujících tabulek
CREATE OR REPLACE TRIGGER trg_mutation_error_demo
    AFTER INSERT ON Guest
    FOR EACH ROW
DECLARE
    v_guest_count NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('MUTATION TRIGGER: Pokus o čtení z mutující tabulky');
    
    -- Tento SELECT způsobí ORA-00001 mutation error
    -- protože čteme z tabulky, která je právě modifikována
    SELECT COUNT(*) INTO v_guest_count FROM Guest;
    
    DBMS_OUTPUT.PUT_LINE('Počet hostů: ' || v_guest_count);
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('CHYBA MUTATION: ' || SQLERRM);
        -- Nepropagujeme chybu dál, jen ji zalogujeme
END;
/

-- =============================================
-- 6. TESTOVÁNÍ TRIGGERŮ
-- =============================================

CREATE OR REPLACE PROCEDURE test_triggers AS
    v_test_guest_id NUMBER;
BEGIN
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    DBMS_OUTPUT.PUT_LINE('TESTOVÁNÍ TRIGGERŮ - LEKCE 13');
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    
    -- Test 1: BEFORE/AFTER triggery
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== TEST 1: BEFORE/AFTER TRIGGERY ===');
    
    BEGIN
        -- Test vložení nového hosta (spustí BEFORE a AFTER INSERT triggery)
        INSERT INTO Guest (firstname, lastname, email, birth_date, street, city, postal_code, country, guest_type)
        VALUES ('Test', 'Trigger', 'TEST.TRIGGER@EXAMPLE.COM', SYSDATE - (25*365), 
                'Test Street 1', 'Test City', '12345', 'Czech Republic', NULL)
        RETURNING guest_id INTO v_test_guest_id;
        
        COMMIT;
        DBMS_OUTPUT.PUT_LINE('Host úspěšně vložen s ID: ' || v_test_guest_id);
        
        -- Test aktualizace (spustí BEFORE a AFTER UPDATE triggery)
        UPDATE Guest 
        SET email = 'updated.email@example.com', guest_type = 'VIP' 
        WHERE guest_id = v_test_guest_id;
        
        COMMIT;
        DBMS_OUTPUT.PUT_LINE('Host úspěšně aktualizován');
        
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Chyba v testu 1: ' || SQLERRM);
            ROLLBACK;
    END;
    
    -- Test 2: INSTEAD OF trigger
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== TEST 2: INSTEAD OF TRIGGER ===');
    
    BEGIN
        -- Vložení do pohledu (spustí INSTEAD OF trigger)
        INSERT INTO v_reservation_details (
            firstname, lastname, email, guest_type,
            room_number, room_type_name, bed_count,
            check_in_date, check_out_date, status, accommodation_price
        ) VALUES (
            'View', 'Test', 'view.test@example.com', 'Business',
            '999', 'Test Suite', 3,
            SYSDATE + 1, SYSDATE + 3, 'Confirmed', 5000
        );
        
        COMMIT;
        DBMS_OUTPUT.PUT_LINE('Rezervace úspěšně vytvořena přes pohled');
        
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Chyba v testu 2: ' || SQLERRM);
            ROLLBACK;
    END;
    
    -- Test 3: Multi-operation trigger
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== TEST 3: MULTI-OPERATION TRIGGER ===');
    
    BEGIN
        -- Test INSERT do Reservation (spustí větvený trigger)
        DECLARE
            v_guest_id NUMBER;
            v_room_id NUMBER;
            v_employee_id NUMBER;
            v_payment_id NUMBER;
        BEGIN
            SELECT guest_id INTO v_guest_id FROM Guest WHERE ROWNUM = 1;
            SELECT room_id INTO v_room_id FROM Room WHERE is_occupied = 0 AND ROWNUM = 1;
            SELECT employee_id INTO v_employee_id FROM Employee WHERE ROWNUM = 1;
            
            INSERT INTO Payment (total_accommodation, total_expenses, is_paid)
            VALUES (3000, 0, 0) RETURNING payment_id INTO v_payment_id;
            
            INSERT INTO Reservation (guest_id, room_id, employee_id, check_in_date, check_out_date, payment_id, status, accommodation_price)
            VALUES (v_guest_id, v_room_id, v_employee_id, SYSDATE + 5, SYSDATE + 7, v_payment_id, 'Confirmed', 3000);
            
            COMMIT;
        END;
        
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Chyba v testu 3: ' || SQLERRM);
            ROLLBACK;
    END;
    
    -- Test 4: Mutation error (očekáváme chybu)
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== TEST 4: MUTATION ERROR TRIGGER ===');
    
    BEGIN
        -- Tento INSERT spustí trigger, který způsobí mutation error
        INSERT INTO Guest (firstname, lastname, email, birth_date, street, city, postal_code, country)
        VALUES ('Mutation', 'Test', 'mutation.test@example.com', SYSDATE - (30*365),
                'Mutation Street', 'Mutation City', '99999', 'Czech Republic');
        
        COMMIT;
        DBMS_OUTPUT.PUT_LINE('Mutation test dokončen (možná s chybou)');
        
    EXCEPTION
        WHEN OTHERS THEN
            DBMS_OUTPUT.PUT_LINE('Očekávaná chyba mutation: ' || SQLERRM);
            ROLLBACK;
    END;
    
    -- Zobrazení statistik
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== STATISTIKY OPERACÍ ===');
    
    FOR rec IN (SELECT * FROM operation_stats ORDER BY table_name) LOOP
        DBMS_OUTPUT.PUT_LINE(rec.table_name || ': INSERT=' || rec.insert_count || 
                           ', UPDATE=' || rec.update_count || ', DELETE=' || rec.delete_count);
    END LOOP;
    
    -- Zobrazení posledních audit záznamů
    DBMS_OUTPUT.PUT_LINE('');
    DBMS_OUTPUT.PUT_LINE('=== POSLEDNÍCH 5 AUDIT ZÁZNAMŮ ===');
    
    FOR rec IN (SELECT * FROM (SELECT * FROM audit_log ORDER BY timestamp DESC) WHERE ROWNUM <= 5) LOOP
        DBMS_OUTPUT.PUT_LINE(TO_CHAR(rec.timestamp, 'HH24:MI:SS') || ' - ' || 
                           rec.table_name || ' ' || rec.operation || ' by ' || rec.user_name);
    END LOOP;
    
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
    DBMS_OUTPUT.PUT_LINE('TESTOVÁNÍ TRIGGERŮ DOKONČENO');
    DBMS_OUTPUT.PUT_LINE('===============================================================================');
END;
/

-- Spuštění testů
BEGIN
    test_triggers;
END;
/
