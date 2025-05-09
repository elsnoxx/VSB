CREATE PROCEDURE change_room
    @reservation_id INT,
    @new_room_id INT
AS
BEGIN
    SET XACT_ABORT ON;
    BEGIN TRANSACTION;

    DECLARE @check_in DATE, @check_out DATE, @room_type_id INT, @price_per_night DECIMAL(10,2), @nights INT, @new_price DECIMAL(10,2);

    -- 1. Načti termíny rezervace
    SELECT @check_in = check_in_date, @check_out = check_out_date
    FROM Reservation
    WHERE reservation_id = @reservation_id;

    -- 2. Ověř dostupnost nového pokoje
    IF EXISTS (
        SELECT 1 FROM Reservation
        WHERE room_id = @new_room_id
          AND status != 'Cancelled'
          AND (
                (@check_in BETWEEN check_in_date AND check_out_date)
             OR (@check_out BETWEEN check_in_date AND check_out_date)
             OR (check_in_date BETWEEN @check_in AND @check_out)
          )
    )
    BEGIN
        ROLLBACK;
        THROW 50001, 'Pokoj není v daném termínu volný', 1;
        RETURN;
    END

    -- 3. Zjisti typ pokoje a cenu za noc
    SELECT @room_type_id = room_type_id FROM Room WHERE room_id = @new_room_id;

    SELECT TOP 1 @price_per_night = price_per_night
    FROM RoomTypePriceHistory
    WHERE room_type_id = @room_type_id
      AND @check_in BETWEEN valid_from AND ISNULL(valid_to, @check_in);

    IF @price_per_night IS NULL
    BEGIN
        ROLLBACK;
        THROW 50005, 'Nebyla nalezena cena za noc pro daný typ pokoje a datum', 1;
        RETURN;
    END

    -- 4. Spočítej novou cenu
    SET @nights = DATEDIFF(DAY, @check_in, @check_out);

    IF @nights < 1
    BEGIN
        ROLLBACK;
        THROW 50002, 'Neplatný počet nocí v rezervaci', 1;
        RETURN;
    END

    SET @new_price = @nights * @price_per_night;

    -- 5. Aktualizuj rezervaci
    UPDATE Reservation
    SET room_id = @new_room_id,
        accommodation_price = @new_price
    WHERE reservation_id = @reservation_id;

    COMMIT;
END