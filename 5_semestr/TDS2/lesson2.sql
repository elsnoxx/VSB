-- • Lesson 2
-- • Add your own variables in the declaration:
--      o Variables of type String, Integer, Date, BOOLEAN
--      o Define a constant
--      o Define a variable with a default value
--      o By transformation using %TYPE
--      o Using implicit explicit data type conversion
--      o Output of results to the console (DBMS.PUT_LINE)

DECLARE
    myString VARCHAR2(100) := 'Hello, My Small World!';
    myInteger INTEGER := 10;
    myDate DATE := SYSDATE;
    myBoolean BOOLEAN := TRUE;

    CONSTANT_VALUE CONSTANT NUMBER := 100;

    defaultValue NUMBER DEFAULT 50;

    salary NUMBER := 80000;
    bonus NUMBER(7, 2);
    bonusPercentage NUMBER(3, 2) := 0.10;

    -- Nested procedure
    PROCEDURE CalculateBonus IS
        calculatedBonus NUMBER;
    BEGIN
        calculatedBonus := salary * bonusPercentage;
        DBMS_OUTPUT.PUT_LINE('Calculated Bonus: ' || calculatedBonus);
    END CalculateBonus;

BEGIN
    myInteger := TO_NUMBER('20');

    -- Output results to the console
    DBMS_OUTPUT.PUT_LINE('String: ' || myString);
    DBMS_OUTPUT.PUT_LINE('Integer: ' || myInteger);
    DBMS_OUTPUT.PUT_LINE('Date: ' || TO_CHAR(myDate, 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Boolean: ' || CASE WHEN myBoolean THEN 'TRUE' ELSE 'FALSE' END);
    DBMS_OUTPUT.PUT_LINE('Constant: ' || CONSTANT_VALUE);
    DBMS_OUTPUT.PUT_LINE('Default Value: ' || defaultValue);

    -- Calling nested procedure
    CalculateBonus;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred: ' || SQLERRM);
END;
