CREATE PROCEDURE change_room
    @reservation_id INT,
    @new_room_id INT
AS
BEGIN
    SET XACT_ABORT ON;
    BEGIN TRANSACTION;

    DECLARE @check_in DATE, @check_out DATE;

    -- 1. Načti termíny rezervace
    SELECT @check_in = check_in_date, @check_out = check_out_date
    FROM Reservation
    WHERE reservation_id = @reservation_id;

    -- 1a. Ověř, že počet dní je platný
    IF DATEDIFF(DAY, @check_in, @check_out) < 1
    BEGIN
        ROLLBACK;
        THROW 50002, 'Neplatný počet dní (check-in musí být před check-out)', 1;
        RETURN;
    END

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

    -- 3. Aktualizuj rezervaci včetně výpočtu ceny v jednom dotazu
    UPDATE r
    SET 
        room_id = @new_room_id,
        accommodation_price = 
            CASE 
                WHEN DATEDIFF(DAY, r.check_in_date, r.check_out_date) < 1 THEN
                    NULL
                ELSE
                    DATEDIFF(DAY, r.check_in_date, r.check_out_date) * 
                    (
                        SELECT TOP 1 price_per_night
                        FROM RoomTypePriceHistory h
                        JOIN Room rm ON rm.room_type_id = h.room_type_id
                        WHERE rm.room_id = @new_room_id
                          AND r.check_in_date BETWEEN h.valid_from AND ISNULL(h.valid_to, r.check_in_date)
                    )
            END
    FROM Reservation r
    WHERE r.reservation_id = @reservation_id;

    -- 4. Ověř, že byla cena nastavena (tj. existuje platná cena a počet nocí je OK)
    IF EXISTS (
        SELECT 1 FROM Reservation
        WHERE reservation_id = @reservation_id
          AND (accommodation_price IS NULL OR accommodation_price <= 0)
    )
    BEGIN
        ROLLBACK;
        THROW 50005, 'Chyba při výpočtu ceny nebo neplatný počet nocí', 1;
        RETURN;
    END

    COMMIT;
END