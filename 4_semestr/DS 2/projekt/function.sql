-- 1. Vytvoření nové rezervace
CREATE OR REPLACE PROCEDURE create_reservation(
    p_guest_id IN NUMBER,
    p_room_type_id IN NUMBER,
    p_employee_id IN NUMBER,
    p_check_in_date IN DATE,
    p_check_out_date IN DATE,
    p_total_accommodation IN NUMBER,
    p_total_expenses IN NUMBER,
    p_result OUT NUMBER
)
AS
    v_room_id NUMBER;
    v_payment_id NUMBER;
    v_is_available NUMBER := 0;
BEGIN
    -- Kontrola platnosti dat
    IF p_check_in_date >= p_check_out_date THEN
        RAISE_APPLICATION_ERROR(-20001, 'Datum příjezdu musí být před datem odjezdu');
    END IF;

    -- Kontrola dostupnosti pokoje
    SELECT COUNT(r.room_id) INTO v_is_available
    FROM Room r
    WHERE r.room_type_id = p_room_type_id
    AND r.is_occupied = 0
    AND r.room_id NOT IN (
        SELECT res.room_id 
        FROM Reservation res 
        WHERE res.status != 'Cancelled'
        AND (
            (p_check_in_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (p_check_out_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (res.check_in_date BETWEEN p_check_in_date AND p_check_out_date)
        )
    )
    AND ROWNUM = 1;
    
    IF v_is_available = 0 THEN
        RAISE_APPLICATION_ERROR(-20002, 'Žádný dostupný pokoj požadovaného typu v daném termínu');
    END IF;
    
    -- Výběr konkrétního pokoje
    SELECT r.room_id INTO v_room_id
    FROM Room r
    WHERE r.room_type_id = p_room_type_id
    AND r.is_occupied = 0
    AND r.room_id NOT IN (
        SELECT res.room_id 
        FROM Reservation res 
        WHERE res.status != 'Cancelled'
        AND (
            (p_check_in_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (p_check_out_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (res.check_in_date BETWEEN p_check_in_date AND p_check_out_date)
        )
    )
    AND ROWNUM = 1;

    -- Zahájení transakce
    SAVEPOINT start_reservation;
    
    -- Vytvoření platby
    INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid)
    VALUES (p_total_accommodation, p_total_expenses, SYSDATE, 0)
    RETURNING payment_id INTO v_payment_id;
    
    -- Vytvoření rezervace
    INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status)
    VALUES (p_guest_id, v_room_id, p_employee_id, SYSDATE, p_check_in_date, p_check_out_date, v_payment_id, 'Confirmed')
    RETURNING reservation_id INTO p_result;
    
    -- Pokud je rezervace na dnešní den, označíme pokoj jako obsazený
    IF p_check_in_date = TRUNC(SYSDATE) THEN
        UPDATE Room SET is_occupied = 1 WHERE room_id = v_room_id;
    END IF;
    
    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK TO start_reservation;
        p_result := -1;
        RAISE;
END;
/

-- 2. Úprava existující rezervace
CREATE OR REPLACE PROCEDURE update_reservation(
    p_reservation_id IN NUMBER,
    p_room_type_id IN NUMBER,
    p_check_in_date IN DATE,
    p_check_out_date IN DATE,
    p_status IN VARCHAR2,
    p_success OUT NUMBER
)
AS
    v_old_room_id NUMBER;
    v_new_room_id NUMBER;
    v_is_available NUMBER := 0;
BEGIN
    -- Kontrola platnosti dat
    IF p_check_in_date >= p_check_out_date THEN
        RAISE_APPLICATION_ERROR(-20001, 'Datum příjezdu musí být před datem odjezdu');
    END IF;
    
    -- Získání aktuální místnosti
    SELECT room_id INTO v_old_room_id
    FROM Reservation
    WHERE reservation_id = p_reservation_id;
    
    -- Kontrola dostupnosti nového pokoje (mimo vlastní rezervaci)
    SELECT COUNT(r.room_id) INTO v_is_available
    FROM Room r
    WHERE r.room_type_id = p_room_type_id
    AND (r.is_occupied = 0 OR r.room_id = v_old_room_id)
    AND r.room_id NOT IN (
        SELECT res.room_id 
        FROM Reservation res 
        WHERE res.status != 'Cancelled'
        AND res.reservation_id != p_reservation_id
        AND (
            (p_check_in_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (p_check_out_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (res.check_in_date BETWEEN p_check_in_date AND p_check_out_date)
        )
    )
    AND ROWNUM = 1;
    
    IF v_is_available = 0 THEN
        RAISE_APPLICATION_ERROR(-20002, 'Žádný dostupný pokoj požadovaného typu v daném termínu');
    END IF;
    
    -- Výběr konkrétního pokoje
    SELECT r.room_id INTO v_new_room_id
    FROM Room r
    WHERE r.room_type_id = p_room_type_id
    AND (r.is_occupied = 0 OR r.room_id = v_old_room_id)
    AND r.room_id NOT IN (
        SELECT res.room_id 
        FROM Reservation res 
        WHERE res.status != 'Cancelled'
        AND res.reservation_id != p_reservation_id
        AND (
            (p_check_in_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (p_check_out_date BETWEEN res.check_in_date AND res.check_out_date)
            OR (res.check_in_date BETWEEN p_check_in_date AND p_check_out_date)
        )
    )
    AND ROWNUM = 1;
    
    -- Zahájení transakce
    SAVEPOINT start_update;
    
    -- Uvolnění starého pokoje, pokud se změnil
    IF v_old_room_id != v_new_room_id THEN
        UPDATE Room SET is_occupied = 0 WHERE room_id = v_old_room_id;
    END IF;
    
    -- Aktualizace rezervace
    UPDATE Reservation 
    SET room_id = v_new_room_id,
        check_in_date = p_check_in_date,
        check_out_date = p_check_out_date,
        status = p_status
    WHERE reservation_id = p_reservation_id;
    
    -- Pokud je rezervace aktivní (Checked In), označíme nový pokoj jako obsazený
    IF p_status = 'Checked In' THEN
        UPDATE Room SET is_occupied = 1 WHERE room_id = v_new_room_id;
    END IF;
    
    COMMIT;
    p_success := 1;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK TO start_update;
        p_success := 0;
        RAISE;
END;
/

-- 3. Zrušení rezervace
CREATE OR REPLACE PROCEDURE cancel_reservation(
    p_reservation_id IN NUMBER,
    p_cancel_reason IN VARCHAR2 DEFAULT NULL
)
AS
    v_room_id NUMBER;
    v_current_status VARCHAR2(20);
BEGIN
    -- Ověření, že rezervace existuje a není již zrušena
    SELECT status, room_id INTO v_current_status, v_room_id
    FROM Reservation
    WHERE reservation_id = p_reservation_id;
    
    IF v_current_status = 'Cancelled' THEN
        RAISE_APPLICATION_ERROR(-20003, 'Rezervace je již zrušena');
    END IF;
    
    -- Zahájení transakce
    SAVEPOINT start_cancel;
    
    -- Aktualizace statusu rezervace
    UPDATE Reservation 
    SET status = 'Cancelled'
    WHERE reservation_id = p_reservation_id;
    
    -- Uložení důvodu zrušení, pokud byl zadán
    IF p_cancel_reason IS NOT NULL THEN
        -- Zde by se důvod zrušení mohl ukládat do samostatné tabulky
        -- nebo přidat sloupec notes do tabulky Reservation
        NULL;
    END IF;
    
    -- Uvolnění pokoje, pokud byl obsazen
    UPDATE Room SET is_occupied = 0 WHERE room_id = v_room_id;
    
    COMMIT;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        ROLLBACK TO start_cancel;
        RAISE_APPLICATION_ERROR(-20004, 'Rezervace s ID ' || p_reservation_id || ' neexistuje');
    WHEN OTHERS THEN
        ROLLBACK TO start_cancel;
        RAISE;
END;
/

-- 4. Zobrazení detailů rezervace
CREATE OR REPLACE FUNCTION get_reservation_details(
    p_reservation_id IN NUMBER
) RETURN SYS_REFCURSOR
AS
    v_result SYS_REFCURSOR;
BEGIN
    OPEN v_result FOR
    SELECT 
        r.reservation_id, r.creation_date, r.check_in_date, r.check_out_date, r.status,
        g.guest_id, g.firstname AS guest_firstname, g.lastname AS guest_lastname,
        g.email AS guest_email, g.phone AS guest_phone, g.guest_type,
        rm.room_id, rm.room_number, rt.name AS room_type, rt.price_per_night,
        e.employee_id, e.firstname AS emp_firstname, e.lastname AS emp_lastname, e.position,
        p.payment_id, p.total_accommodation, p.total_expenses, p.is_paid, p.payment_date,
        (p.total_accommodation + p.total_expenses) AS total_amount,
        a.street, a.city, a.postal_code, a.country
    FROM Reservation r
    JOIN Guest g ON r.guest_id = g.guest_id
    JOIN Room rm ON r.room_id = rm.room_id
    JOIN RoomType rt ON rm.room_type_id = rt.room_type_id
    JOIN Employee e ON r.employee_id = e.employee_id
    JOIN Payment p ON r.payment_id = p.payment_id
    JOIN Address a ON g.address_id = a.address_id
    WHERE r.reservation_id = p_reservation_id;
    
    RETURN v_result;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RAISE_APPLICATION_ERROR(-20005, 'Rezervace s ID ' || p_reservation_id || ' neexistuje');
    WHEN OTHERS THEN
        RAISE;
END;
/

-- 5. Přidání platby k rezervaci
CREATE OR REPLACE PROCEDURE add_payment_to_reservation(
    p_reservation_id IN NUMBER,
    p_total_accommodation IN NUMBER,
    p_total_expenses IN NUMBER,
    p_is_paid IN NUMBER
)
AS
    v_payment_id NUMBER;
    v_old_payment_id NUMBER;
BEGIN
    -- Získání ID stávající platby
    SELECT payment_id INTO v_old_payment_id
    FROM Reservation
    WHERE reservation_id = p_reservation_id;
    
    -- Zahájení transakce
    SAVEPOINT start_payment;
    
    -- Vytvoření nové platby
    INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid)
    VALUES (p_total_accommodation, p_total_expenses, SYSDATE, p_is_paid)
    RETURNING payment_id INTO v_payment_id;
    
    -- Aktualizace rezervace s novou platbou
    UPDATE Reservation SET payment_id = v_payment_id
    WHERE reservation_id = p_reservation_id;
    
    COMMIT;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        ROLLBACK TO start_payment;
        RAISE_APPLICATION_ERROR(-20006, 'Rezervace s ID ' || p_reservation_id || ' neexistuje');
    WHEN OTHERS THEN
        ROLLBACK TO start_payment;
        RAISE;
END;
/

-- 6. Úprava údajů o hostovi
CREATE OR REPLACE PROCEDURE update_guest_info(
    p_guest_id IN NUMBER,
    p_firstname IN VARCHAR2,
    p_lastname IN VARCHAR2,
    p_email IN VARCHAR2,
    p_phone IN VARCHAR2,
    p_street IN VARCHAR2,
    p_city IN VARCHAR2,
    p_postal_code IN VARCHAR2,
    p_country IN VARCHAR2
)
AS
    v_address_id NUMBER;
BEGIN
    -- Získání ID adresy hosta
    SELECT address_id INTO v_address_id
    FROM Guest
    WHERE guest_id = p_guest_id;
    
    -- Zahájení transakce
    SAVEPOINT start_update_guest;
    
    -- Aktualizace údajů o hostovi
    UPDATE Guest
    SET firstname = p_firstname,
        lastname = p_lastname,
        email = p_email,
        phone = p_phone
    WHERE guest_id = p_guest_id;
    
    -- Aktualizace adresy
    UPDATE Address
    SET street = p_street,
        city = p_city,
        postal_code = p_postal_code,
        country = p_country
    WHERE address_id = v_address_id;
    
    COMMIT;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        ROLLBACK TO start_update_guest;
        RAISE_APPLICATION_ERROR(-20007, 'Host s ID ' || p_guest_id || ' neexistuje');
    WHEN OTHERS THEN
        ROLLBACK TO start_update_guest;
        RAISE;
END;
/

-- 7. Změna statusu rezervace
CREATE OR REPLACE PROCEDURE change_reservation_status(
    p_reservation_id IN NUMBER,
    p_new_status IN VARCHAR2
)
AS
    v_room_id NUMBER;
    v_current_status VARCHAR2(20);
BEGIN
    -- Kontrola platnosti nového statusu
    IF p_new_status NOT IN ('Confirmed', 'Checked In', 'Checked Out', 'Cancelled') THEN
        RAISE_APPLICATION_ERROR(-20008, 'Neplatný status rezervace');
    END IF;
    
    -- Získání aktuálního statusu a ID pokoje
    SELECT status, room_id INTO v_current_status, v_room_id
    FROM Reservation
    WHERE reservation_id = p_reservation_id;
    
    -- Zahájení transakce
    SAVEPOINT start_status_change;
    
    -- Aktualizace statusu rezervace
    UPDATE Reservation SET status = p_new_status
    WHERE reservation_id = p_reservation_id;
    
    -- Aktualizace stavu pokoje podle nového statusu
    IF p_new_status = 'Checked In' THEN
        UPDATE Room SET is_occupied = 1 WHERE room_id = v_room_id;
    ELSIF p_new_status IN ('Checked Out', 'Cancelled') THEN
        UPDATE Room SET is_occupied = 0 WHERE room_id = v_room_id;
    END IF;
    
    COMMIT;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        ROLLBACK TO start_status_change;
        RAISE_APPLICATION_ERROR(-20009, 'Rezervace s ID ' || p_reservation_id || ' neexistuje');
    WHEN OTHERS THEN
        ROLLBACK TO start_status_change;
        RAISE;
END;
/

-- 8. Zobrazení seznamu rezervací podle data
CREATE OR REPLACE FUNCTION get_reservations_by_date(
    p_date_type IN VARCHAR2,  -- 'check_in', 'check_out', 'creation'
    p_start_date IN DATE,
    p_end_date IN DATE
) RETURN SYS_REFCURSOR
AS
    v_result SYS_REFCURSOR;
    v_query VARCHAR2(1000);
BEGIN
    -- Vytvoření dynamického dotazu podle typu datumu
    v_query := 'SELECT r.reservation_id, r.check_in_date, r.check_out_date, 
                       g.firstname || '' '' || g.lastname AS guest_name,
                       rm.room_number, r.status, p.is_paid,
                       p.total_accommodation + p.total_expenses AS total_amount
                FROM Reservation r
                JOIN Guest g ON r.guest_id = g.guest_id
                JOIN Room rm ON r.room_id = rm.room_id
                JOIN Payment p ON r.payment_id = p.payment_id
                WHERE r.';
                
    IF p_date_type = 'check_in' THEN
        v_query := v_query || 'check_in_date BETWEEN :start_date AND :end_date';
    ELSIF p_date_type = 'check_out' THEN
        v_query := v_query || 'check_out_date BETWEEN :start_date AND :end_date';
    ELSIF p_date_type = 'creation' THEN
        v_query := v_query || 'creation_date BETWEEN :start_date AND :end_date';
    ELSE
        RAISE_APPLICATION_ERROR(-20010, 'Neplatný typ datumu: ' || p_date_type);
    END IF;
    
    v_query := v_query || ' ORDER BY r.' || p_date_type || '_date';
    
    -- Otevření kurzoru s dynamickým dotazem
    OPEN v_result FOR v_query USING p_start_date, p_end_date;
    
    RETURN v_result;
EXCEPTION
    WHEN OTHERS THEN
        RAISE;
END;
/

-- 9. Označení pokoje jako obsazeného/volného
CREATE OR REPLACE PROCEDURE update_room_occupancy(
    p_room_id IN NUMBER,
    p_is_occupied IN NUMBER
)
AS
BEGIN
    -- Ověření, že pokoj existuje
    DECLARE
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count FROM Room WHERE room_id = p_room_id;
        IF v_count = 0 THEN
            RAISE_APPLICATION_ERROR(-20011, 'Pokoj s ID ' || p_room_id || ' neexistuje');
        END IF;
    END;
    
    -- Aktualizace stavu pokoje
    UPDATE Room 
    SET is_occupied = p_is_occupied
    WHERE room_id = p_room_id;
    
    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/

-- 10. Přidání nového hosta do systému
CREATE OR REPLACE FUNCTION add_new_guest(
    p_firstname IN VARCHAR2,
    p_lastname IN VARCHAR2,
    p_email IN VARCHAR2,
    p_phone IN VARCHAR2,
    p_birth_date IN DATE,
    p_guest_type IN VARCHAR2,
    p_street IN VARCHAR2,
    p_city IN VARCHAR2,
    p_postal_code IN VARCHAR2,
    p_country IN VARCHAR2,
    p_notes IN VARCHAR2 DEFAULT NULL
) RETURN NUMBER
AS
    v_address_id NUMBER;
    v_guest_id NUMBER;
BEGIN
    -- Kontrola platnosti typu hosta
    IF p_guest_type NOT IN ('Regular', 'VIP', 'Loyalty') THEN
        RAISE_APPLICATION_ERROR(-20012, 'Neplatný typ hosta');
    END IF;
    
    -- Kontrola duplicity emailu
    DECLARE
        v_count NUMBER;
    BEGIN
        SELECT COUNT(*) INTO v_count FROM Guest WHERE email = p_email;
        IF v_count > 0 THEN
            RAISE_APPLICATION_ERROR(-20013, 'Host s emailem ' || p_email || ' již existuje');
        END IF;
    END;
    
    -- Zahájení transakce
    SAVEPOINT start_add_guest;
    
    -- Vložení nové adresy
    INSERT INTO Address (street, city, postal_code, country)
    VALUES (p_street, p_city, p_postal_code, p_country)
    RETURNING address_id INTO v_address_id;
    
    -- Vložení nového hosta
    INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes)
    VALUES (p_firstname, p_lastname, p_email, p_phone, p_birth_date, v_address_id, p_guest_type, SYSDATE, p_notes)
    RETURNING guest_id INTO v_guest_id;
    
    COMMIT;
    RETURN v_guest_id;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK TO start_add_guest;
        RAISE;
END;
/