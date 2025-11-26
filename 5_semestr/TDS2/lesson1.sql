-- Lesson 1
--      o Anonymous procedure – containing:
--      o about the Declaration
--      o Block
--      o Exception

-- select * from PAYMENT where PAYMENT_ID = 1;

DECLARE
    payment_id NUMBER := 3;
    payment_price NUMBER;
    payment_discount FLOAT := 0.1;
    payment_price_discounted NUMBER;

BEGIN
    SELECT TOTAL_EXPENSES + TOTAL_ACCOMMODATION
    INTO payment_price
    FROM PAYMENT
    WHERE PAYMENT_ID = 2;


    IF payment_price IS NOT NULL THEN
        payment_price_discounted := payment_price * (1 - payment_discount);
        DBMS_OUTPUT.PUT_LINE('Discount for payment ' || payment_price ||
                             ' is: ' || payment_price_discounted);
    ELSE
        DBMS_OUTPUT.PUT_LINE('No payment data for payment id: ' || payment_id || ' found');
    END IF;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        DBMS_OUTPUT.PUT_LINE('No data found for payment ' || payment_id);
    WHEN TOO_MANY_ROWS THEN
        DBMS_OUTPUT.PUT_LINE('More than one row found for payment ' || payment_id);
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred: ' || SQLERRM);
END;
