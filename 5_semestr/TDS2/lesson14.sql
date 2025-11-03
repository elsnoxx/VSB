-- ===============================================
-- TDS2 - Lekce 11
-- ===============================================


-- Prozkoumejte závislosti mezi objekty, například vzájemně se volající procedury,
-- pohledy z tabulek atd.

CREATE OR REPLACE PROCEDURE proc_a IS
BEGIN
  DBMS_OUTPUT.PUT_LINE('Procedura A');
END;
/

CREATE OR REPLACE PROCEDURE proc_b IS
BEGIN
  proc_a; -- volá proceduru A
  DBMS_OUTPUT.PUT_LINE('Procedura B');
END;
/

SELECT name, referenced_name, referenced_type
FROM user_dependencies
WHERE name = 'PROC_B';


-- Vypište stroj závislosti, například z tabulky, je vytvořen pohled, který je používám v
-- jedné proceduře a ta je následně používaná v druhé proceduře

SELECT LPAD(' ', 2*(LEVEL-1)) || name AS object_name,
       type,
       referenced_name,
       referenced_type
FROM user_dependencies
START WITH name = 'PROC_B'
CONNECT BY PRIOR referenced_name = name;
