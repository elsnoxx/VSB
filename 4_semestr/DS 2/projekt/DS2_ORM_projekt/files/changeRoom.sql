CREATE OR REPLACE PROCEDURE change_room(
    p_reservation_id IN NUMBER,
    p_new_room_id IN NUMBER
) AS
    v_count NUMBER;
    v_room_type_id NUMBER;
    v_check_in_date DATE;
    v_check_out_date DATE;
    v_price_per_night NUMBER;
    v_nights NUMBER;
    v_new_price NUMBER;
BEGIN
    -- Zjisti termín rezervace
    SELECT check_in_date, check_out_date INTO v_check_in_date, v_check_out_date
    FROM Reservation WHERE reservation_id = p_reservation_id;

    -- Zkontroluj kolize
    SELECT COUNT(*) INTO v_count FROM Reservation
    WHERE room_id = p_new_room_id
      AND status != 'Cancelled'
      AND (
        (v_check_in_date BETWEEN check_in_date AND check_out_date)
        OR (v_check_out_date BETWEEN check_in_date AND check_out_date)
        OR (check_in_date BETWEEN v_check_in_date AND v_check_out_date)
      );

    IF v_count > 0 THEN
        RAISE_APPLICATION_ERROR(-20001, 'Pokoj není v daném termínu volný');
    END IF;

    -- Zjisti typ pokoje
    SELECT room_type_id INTO v_room_type_id FROM Room WHERE room_id = p_new_room_id;

    -- Zjisti cenu za noc
    SELECT price_per_night INTO v_price_per_night
    FROM RoomTypePriceHistory
    WHERE room_type_id = v_room_type_id
      AND v_check_in_date BETWEEN valid_from AND NVL(valid_to, v_check_in_date);

    -- Spočítej počet nocí
    v_nights := v_check_out_date - v_check_in_date;

    -- Spočítej novou cenu
    v_new_price := v_nights * v_price_per_night;

    -- Aktualizuj rezervaci
    UPDATE Reservation
    SET room_id = p_new_room_id,
        accommodation_price = v_new_price
    WHERE reservation_id = p_reservation_id;

    COMMIT;
END;
/