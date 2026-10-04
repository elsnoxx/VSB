-- ===============================================
-- TDS2 - Lekce 14
-- ===============================================

-- 1) Prozkoumání závislostí mezi objekty
CREATE OR REPLACE PROCEDURE proc_a IS
BEGIN
  DBMS_OUTPUT.PUT_LINE('Procedura A');
END;
/

CREATE OR REPLACE PROCEDURE proc_b IS
BEGIN
  proc_a;
  DBMS_OUTPUT.PUT_LINE('Procedura B');
END;
/

SELECT name, referenced_name, referenced_type
FROM user_dependencies
WHERE name = 'PROC_B';

-- 2) Strom závislostí
SELECT LPAD(' ', 2*(LEVEL-1)) || name AS object_name,
       type,
       referenced_name,
       referenced_type
FROM user_dependencies
START WITH name = 'PROC_B'
CONNECT BY PRIOR referenced_name = name;

-- 3) TIMESTAMP a SIGNATURE ukázka
SELECT name, referenced_name, timestamp, referenced_timestamp
FROM user_dependencies
WHERE name = 'PROC_B';

SELECT name, referenced_name, signature
FROM user_dependencies
WHERE name = 'PROC_B';

-- Pro přepnutí režimu SIGNATURE (stačí ukázat)
-- ALTER SESSION SET REMOTE_DEPENDENCIES_MODE = SIGNATURE;

