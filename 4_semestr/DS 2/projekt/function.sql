CREATE OR REPLACE PROCEDURE create_reservation(
    p_room_id         IN NUMBER,
    p_guest_id        IN NUMBER,
    p_employee_id     IN NUMBER,
    p_check_in_date   IN DATE,
    p_check_out_date  IN DATE
) AS
    v_room_count         NUMBER;
    v_guest_count        NUMBER;
    v_room_type_id       NUMBER;
    v_price_per_night    NUMBER;
    v_nights             NUMBER;
    v_total_accommodation NUMBER;
    v_payment_id         NUMBER;
BEGIN
    -- Kontrola dostupnosti pokoje
    SELECT COUNT(*)
      INTO v_room_count
      FROM Reservation res
     WHERE res.room_id = p_room_id
       AND res.status != 'Cancelled'
       AND (
            (p_check_in_date BETWEEN res.check_in_date AND res.check_out_date)
         OR (p_check_out_date BETWEEN res.check_in_date AND res.check_out_date)
         OR (res.check_in_date BETWEEN p_check_in_date AND p_check_out_date)
       );

    IF v_room_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'Pokoj je v daném termínu již obsazený');
    END IF;

    -- Kontrola platnosti datumů
    IF p_check_in_date >= p_check_out_date THEN
        RAISE_APPLICATION_ERROR(-20002, 'Datum příjezdu musí být před datem odjezdu');
    END IF;

    -- Ověření existence hosta
    SELECT COUNT(*)
      INTO v_guest_count
      FROM Guest
     WHERE guest_id = p_guest_id;

    IF v_guest_count = 0 THEN
        RAISE_APPLICATION_ERROR(-20003, 'Host v systému neexistuje');
    END IF;

    -- Získání typu pokoje a ceny za noc
    SELECT rt.room_type_id, rt.price_per_night
      INTO v_room_type_id, v_price_per_night
      FROM Room r
      JOIN RoomType rt ON r.room_type_id = rt.room_type_id
     WHERE r.room_id = p_room_id;

    -- Výpočet délky pobytu a celkové ceny
    v_nights := p_check_out_date - p_check_in_date;
    v_total_accommodation := v_nights * v_price_per_night;

    -- Vytvoření platby (nezaplacené)
    INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid)
         VALUES (v_total_accommodation, v_total_accommodation, NULL, 0)
      RETURNING payment_id INTO v_payment_id;

    -- Vytvoření rezervace
    INSERT INTO Reservation (
        guest_id, room_id, employee_id, creation_date,
        check_in_date, check_out_date, payment_id, status
    ) VALUES (
        p_guest_id, p_room_id, p_employee_id, SYSDATE,
        p_check_in_date, p_check_out_date, v_payment_id, 'Confirmed'
    );

    -- Aktualizace stavu pokoje
    UPDATE Room
       SET is_occupied = 1
     WHERE room_id = p_room_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END create_reservation;
