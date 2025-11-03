-- ===============================================
-- TDS2 - Lekce 11
-- ===============================================



-- Vyzkoušejte optimalizační parametry Oracle, PLSQL_CODE_TYPE,
-- PLSQL_OPTIMIZATIUON_LEVEL, a vypište jejich nastavení z pohledu
-- USER_PLSQL_OBJECT_SETTING

SHOW PARAMETER PLSQL;

ALTER SESSION SET PLSQL_CODE_TYPE = NATIVE;
ALTER SESSION SET PLSQL_OPTIMIZATION_LEVEL = 3;


CREATE OR REPLACE PROCEDURE demo_proc IS
BEGIN
  DBMS_OUTPUT.PUT_LINE('Optimalizace test');
END;
/

ALTER PROCEDURE demo_proc COMPILE;

SELECT *
FROM USER_PLSQL_OBJECT_SETTINGS
WHERE name = 'DEMO_PROC';


-- Vyzkoušejte různé úrovně optimalizace v závislosti na rychlosti provádění složité
-- procedury

-- Vypnutí výpisu
SET SERVEROUTPUT ON;

-- 1️⃣ Vytvoření testovací tabulky a procedury
DROP TABLE test_opt PURGE;

CREATE TABLE test_opt AS
SELECT LEVEL AS id, DBMS_RANDOM.VALUE(1,1000) AS num
FROM dual
CONNECT BY LEVEL <= 100000;

CREATE OR REPLACE PROCEDURE heavy_proc IS
  v_sum NUMBER := 0;
BEGIN
  FOR r IN (SELECT num FROM test_opt)
  LOOP
    -- Umělá zátěž: složitý výpočet
    v_sum := v_sum + POWER(r.num, 1.5) / (r.num + 1);
  END LOOP;
  DBMS_OUTPUT.PUT_LINE('SUM = ' || v_sum);
END;

DECLARE
  t_start NUMBER;
  t_end NUMBER;
BEGIN
  t_start := DBMS_UTILITY.GET_TIME;

  heavy_proc; -- zavolání procedury

  t_end := DBMS_UTILITY.GET_TIME;
  DBMS_OUTPUT.PUT_LINE('Trvání: ' || TO_CHAR((t_end - t_start)/100) || ' sekund');
END;
/


ALTER SESSION SET PLSQL_OPTIMIZATION_LEVEL = 0;
ALTER PROCEDURE heavy_proc COMPILE;
EXEC heavy_proc;

ALTER SESSION SET PLSQL_OPTIMIZATION_LEVEL = 1;
ALTER PROCEDURE heavy_proc COMPILE;
EXEC heavy_proc;

ALTER SESSION SET PLSQL_OPTIMIZATION_LEVEL = 2;
ALTER PROCEDURE heavy_proc COMPILE;
EXEC heavy_proc;

ALTER SESSION SET PLSQL_OPTIMIZATION_LEVEL = 3;
ALTER PROCEDURE heavy_proc COMPILE;
EXEC heavy_proc;

-- Vyzkoušejte výpis varování kompilátoru PSQL_WARNINGS, DBMS_WARNINGS
SHOW PARAMETER plsql_warnings;

ALTER SESSION SET PLSQL_WARNINGS = 'ENABLE:ALL';

CREATE OR REPLACE PROCEDURE warn_test IS
  v_x NUMBER;   -- nepoužitá proměnná
  v_y NUMBER := 10;
BEGIN
  DBMS_OUTPUT.PUT_LINE('Varování test.');
END;
/

SELECT * 
FROM USER_ERRORS 
WHERE NAME = 'WARN_TEST';


-- Vyzkoušejte podmíněnou kompilace v závislosti na verzi Oracle SQL Serveru,
-- DBMS_DB_VERSION, $IF $END

SET SERVEROUTPUT ON;

CREATE OR REPLACE PROCEDURE test_compile IS
BEGIN
  $IF DBMS_DB_VERSION.VERSION >= 19 $THEN
    DBMS_OUTPUT.PUT_LINE('Kompilováno pro Oracle 19c nebo novější.');
  $ELSE
    DBMS_OUTPUT.PUT_LINE('Kompilováno pro starší verzi Oracle.');
  $END
END;
/


-- Zkuste skrýt pro jednu vámi vybranou proceduru její zdrojový kód
-- DBMS_DDL.CREATE_WRAPPED

CREATE OR REPLACE PROCEDURE my_secret_proc IS
BEGIN
  DBMS_OUTPUT.PUT_LINE('Tajný výpočet');
END;
/


SELECT text
FROM user_source
WHERE name = 'MY_SECRET_PROC'
ORDER BY line;

BEGIN
  DBMS_DDL.CREATE_WRAPPED(
    ddl => q'[
      CREATE OR REPLACE PROCEDURE my_secret_proc IS
      BEGIN
        DBMS_OUTPUT.PUT_LINE('Tajný výpočet – verze WRAPPED');
      END;
    ]'
  );
END;
/
