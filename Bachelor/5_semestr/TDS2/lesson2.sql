-- • Lesson 2
-- • Add your own variables in the declaration:
--      o Variables of type String, Integer, Date, BOOLEAN
--      o Define a constant
--      o Define a variable with a default value
--      o By transformation using %TYPE
--      o Using implicit explicit data type conversion
--      o Output of results to the console (DBMS.PUT_LINE)

DECLARE
    -- 1. Proměnné základních typů (String, Integer, Date, BOOLEAN)
    myString VARCHAR2(100) := 'Hello, My Small World!';
    myInteger INTEGER := 10;
    myDate DATE := SYSDATE;
    myBoolean BOOLEAN := TRUE;

    -- 2. Definování konstanty
    CONSTANT_VALUE CONSTANT NUMBER := 100;

    -- 3. Proměnná s defaultní hodnotou
    defaultValue NUMBER DEFAULT 50;

    -- 4. Proměnné s použitím %TYPE (TOTO VÁM CHYBĚLO!)
    salary NUMBER := 80000;
    employeeSalary salary%TYPE;  -- Převezme datový typ z proměnné salary
    bonus salary%TYPE;          -- Převezme datový typ z proměnné salary
    bonusPercentage NUMBER(3, 2) := 0.10;
    currentPercentage bonusPercentage%TYPE := 0.15; -- Převezme typ z bonusPercentage

    -- Nested procedure (vnořená procedura)
    PROCEDURE CalculateBonus IS
        calculatedBonus NUMBER;
    BEGIN
        calculatedBonus := salary * bonusPercentage;
        DBMS_OUTPUT.PUT_LINE('Calculated Bonus: ' || calculatedBonus);
    END CalculateBonus;

BEGIN
    -- 5. Implicitní/explicitní konverze datových typů
    myInteger := TO_NUMBER('20');  -- Explicitní konverze
    employeeSalary := salary;      -- Implicitní konverze (stejný typ díky %TYPE)
    bonus := salary * 0.2;        -- Implicitní konverze

    -- 6. Výpis výsledků na konzolu (DBMS_OUTPUT.PUT_LINE)
    DBMS_OUTPUT.PUT_LINE('String: ' || myString);
    DBMS_OUTPUT.PUT_LINE('Integer: ' || myInteger);
    DBMS_OUTPUT.PUT_LINE('Date: ' || TO_CHAR(myDate, 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Boolean: ' || CASE WHEN myBoolean THEN 'TRUE' ELSE 'FALSE' END);
    DBMS_OUTPUT.PUT_LINE('Constant: ' || CONSTANT_VALUE);
    DBMS_OUTPUT.PUT_LINE('Default Value: ' || defaultValue);
    DBMS_OUTPUT.PUT_LINE('Employee Salary (%TYPE): ' || employeeSalary);
    DBMS_OUTPUT.PUT_LINE('Bonus (%TYPE): ' || bonus);
    DBMS_OUTPUT.PUT_LINE('Current Percentage (%TYPE): ' || currentPercentage);

    -- 7. Volání vnořené procedury
    CalculateBonus;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('An unexpected error occurred: ' || SQLERRM);
END;
/