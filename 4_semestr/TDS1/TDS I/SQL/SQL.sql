
/*	DD S15 L01
o	Napište dotaz pro spojování øetìzcù pomocí || , pomocí CONCAT()*/

SELECT Name || ' ' || Surname AS full_name
FROM EMPLOYEE
WHERE EMPLOYEE_ID = 1

SELECT CONCAT(CONCAT(Surname, '''s job category is '),
      jobdesc) "Job" 
   FROM employee 
   WHERE employee_id = 1;

/*SELECT DISTINCT 
DD S16 L02*/

SELECT COUNT(DISTINCT Training_ID) AS Pocet_Kategorii_Skoleni
FROM Training

SELECT DISTINCT Employee_ID, Name || ' ' || Surname AS full_name
FROM Employee

/*	WHERE podmínky pro výbìr øádkù
	Funkce LOWER, UPPER, INITCAP */

SELECT
  employee_id, name, surname, email
FROM
  employee
WHERE
  UPPER(surname) = 'Kraninger'; 
    
SELECT
  employee_id, name, surname, email
FROM
  employee
WHERE
  lower(email) = 'ellen.kacalova@msa.com'; 

SELECT INITCAP(email)
FROM EMPLOYEE


/*•	DD S16 L03
o	BETWEEN … AND
o	LIKE (%, _)
o	IN()
o	IS NULL, IS NOT NULL*/


SELECT *
FROM EMPLOYEE JOIN SALARY_CHANGE ON employee.employee_id = salary_change.employee_id
WHERE salary_change.salary BETWEEN 10000 AND 50000;

SELECT *
FROM EMPLOYEE 
WHERE SURNAME LIKE 'K%'

SELECT *
FROM EMPLOYEE 
WHERE Employee_id IN (select employee_id from EMPLOYEE)

SELECT *
FROM EMPLOYEE 
WHERE EoEmployment IS NULL


/* 	DD S17 L01
o	AND, OR, NOT
o	Priorita vyhodnocení pomocí () */

SELECT *
FROM EMPLOYEE 
WHERE (EoEmployment IS NULL) AND SURNAME LIKE 'K%'

SELECT *
FROM EMPLOYEE JOIN SALARY_CHANGE ON employee.employee_id = salary_change.employee_id
WHERE SURNAME LIKE 'K%' OR  salary_change.salary > 50000;

SELECT  Employee_id,Name || ' ' || Surname AS full_name
FROM Employee
WHERE surname NOT LIKE 'S%';

/*•	DD S17 L02
o	ORDER BY atr [ASC/DESC]*/
o	Tøídìní podle jednoho nebo vice atributù */

SELECT Employee_id, name,surname
FROM Employee
WHERE SURNAME LIKE 'K%'
ORDER BY SURNAME DESC

SELECT Employee_id, name,surname
FROM Employee
ORDER BY name ASC

SELECT Employee_id,surname, name
FROM Employee
ORDER BY name ASC,surname

/*o	Jednoøádkové funkce
o	víceøádkové funkce MIN, MAX, AVG, SUM, COUNT */


SELECT MAX(salary)
FROM salary_change

SELECT MIN(salary)
FROM salary_change

SELECT AVG(salary)
FROM salary_change

SELECT SUM(salary)
FROM salary_change

SELECT COUNT(salary)
FROM salary_change


/*víceøádkové funkce MIN, MAX, AVG, SUM, COUNT*/

SELECT Employee_id, MAX(salary)
FROM  Salary_change 
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

SELECT Employee_id, MIN(salary)
FROM  Salary_change 
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

SELECT Employee_id, AVG(salary)
FROM  Salary_change 
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

SELECT Employee_id, SUM(salary)
FROM  Salary_change 
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

SELECT Employee_id, COUNT(salary)
FROM  Salary_change 
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

SELECT Employee_id, MAX(salary)
FROM  Salary_change
GROUP BY employee_id
HAVING COUNT(*)>0
ORDER BY employee_id

/*	SQL S01 L01
o	LOWER, UPPER, INITCAP
o	CONCAT, SUBSTR, LENGTH, INSTR, LPAD, RPAD, TRIM, REPLACE */


SELECT SUBSTR(Surname,0,10) 
FROM Employee

SELECT Surname,LENGTH(Surname)
FROM Employee

SELECT INSTR(Surname,'K', 1, 1)
FROM Employee
WHERE employee_id = 1

SELECT LPAD('Page 1',15,'*.') "LPAD example"
   FROM DUAL;

SELECT LPAD(Surname,(LENGTH(Surname)+9),'Employee ') 
FROM Employee;

SELECT RPAD('Pokus ',(6+LENGTH(NAME)),NAME ) 
FROM Employee;

SELECT TRIM(LEADING 'K' FROM surname) 
FROM Employee
WHERE employee_id = 1

SELECT REPLACE(surname,'Kraning','FUCKING') 
FROM Employee
WHERE employee_id = 1

/*o	Použijte tabulku DUAL
o	SQL S01 L02
•	SQL S01 L02
o	ROUND, TRUNC zaukrouhlení na 2 desetinná místa, na celé tisíce MOD
MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY, ROUND, TRUNC*/

SELECT ROUND(15.193,2) 
FROM DUAL;

SELECT TRUNC(1500.79,2) 
FROM DUAL;

SELECT
round((20355/1000),0)
FROM DUAL;

SELECT TRUNC(1500.79,0) 
FROM DUAL;

/*o	MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY, ROUND, TRUNC
o	Systémová konstanta SYSDATE*/

SELECT MONTHS_BETWEEN 
   (TO_DATE('09-06-2022','MM-DD-YYYY'),
    SYSDATE)
    FROM DUAL

SELECT NEXT_DAY(SYSDATE,6) "NEXT DAY"
FROM DUAL;


SELECT SYSDATE,
   LAST_DAY(SYSDATE) "Last",
   LAST_DAY(SYSDATE) - SYSDATE "Days Left"
   FROM DUAL;

SELECT ROUND (SYSDATE,'YEAR')
   "New Year" FROM DUAL;
   
SELECT TRUNC(SYSDATE, 'MONTH')
  "New month" FROM DUAL;

/*SQL S02 L01
o	TO_CHAR, TO_NUMBER, TO_DATE*/

SELECT TO_CHAR(MONTHS_BETWEEN 
   (TO_DATE('09-06-2022','MM-DD-YYYY'),
    SYSDATE))
    FROM DUAL

SELECT TO_DATE(
    'January 15, 1989, 11:00 A.M.',
    'Month dd, YYYY, HH:MI A.M.',
     'NLS_DATE_LANGUAGE = American')
     FROM DUAL;
     
SELECT
TO_NUMBER('4687841', '9999999')
FROM DUAL;

/*•	SQL S02 L02
o	NVL, NVL2, NULLIF, COALESCE*/

SELECT nvl(Hobbies,'Missing information on someone`s hobbies ')
FROM PI

SELECT NVL2(Check_end,'DONE', 'MISSING') STATUS
FROM CHECKS

SELECT NULLIF(NAME, SURNAME)
FROM Employee

SELECT Hobbies, COALESCE(Hobbies, 'UNKNOWN') AS Hobby
FROM PI

/*•	SQL S02 L03
o	DECODE, CASE, IF-THEN-ELSE*/

SELECT Employee_ID,
DECODE(Employee_ID, 1, 'MAJKL',
                    2, 'Laska',
                    'Nikdo dulezity') result
FROM Employee;


SELECT  salary, 
   (CASE  
   WHEN salary<15000 THEN 'Low'
   WHEN salary >100000 THEN 'High'
   ELSE 'Medium' 
   END) salary   
FROM    Salary_Change

/*•	SQL S03 L01
o	NATURAL JOIN, CROSS JOIN*/

SELECT Employee_ID, name, surname, salary
FROM Employee
NATURAL JOIN Salary_change

SELECT surname, salary
FROM Employee
CROSS JOIN Salary_change;

/*	SQL S03 L02
o	JOIN … USING(atr)
o	JOIN .. ON (podmínka spojení) */

SELECT Employee_ID, name, surname, salary
FROM Employee 
JOIN Salary_change USING (Employee_id);

SELECT e.Employee_ID, name, surname, salary
FROM Employee e
JOIN Salary_change s ON e.Employee_ID = s.Employee_Id


/*•	SQL S03 L03
o	LEFT OUTER JOIN … ON ()
o	RIGHT OUTER JOIN … ON ()
o	FULL OUTER JOIN … ON () */

SELECT e.Employee_ID, name, surname, salary
FROM Employee e
LEFT OUTER JOIN Salary_change s ON e.Employee_ID = s.Employee_Id

SELECT e.Employee_ID, name, surname, salary
FROM Employee e
RIGHT OUTER JOIN Salary_change s ON e.Employee_ID = s.Employee_Id

SELECT e.Employee_ID, name, surname, salary
FROM Employee e
FULL OUTER JOIN Salary_change s ON e.Employee_ID = s.Employee_Id

/*•	SQL S03 L04
o	Spojování 2x stejné tabulky s pøejmenováním (vazba mezi nadøízenými a podøízenými v jedné tabulce)
o	Hierarchické dotazování – stromová struktura zanoøení START WITH, CONNECT BY PRIOR, LEVEL

SELECT Employee_id,Surname, NAME, Manager_id
FROM Employee
/*where level= 2*/ 
START WITH Manager_ID = NULL
CONNECT BY Manager_id = prior employee_id
Order by employee_id

SELECT LPAD(surname, lENGTH(surname)+(LEVEL*2)-2,'_') AS "ORGANISATIONAL DIAGRAM"
FROM Employee
/*WHERE LEVEL =2*/
WHERE surname != 'Kraninger'
START WITH manager_id is NULL CONNECT BY PRIOR employeE_id = manager_id
Order by employee_id

SELECT W.surname || ' works for ' || M.surname
AS "Works for"
FROM employee W JOIN employee M
ON (W.manager_id = M.employee_id)

SELECT W.surname, W.manager_id, M.surname
AS "Manager name"
FROM employee W JOIN employee M
ON (W.manager_id = M.employee_id);

/*•	SQL S04 L02
o	AVG, COUNT, MIN, MAX, SUM, VARIANCE, STDDEV*/


SELECT Employee_id, surname, name, AVG(salary)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY Employee_id,surname, name
ORDER BY Employee_id


SELECT Employee_id, surname, count(RoCH_ID)
FROM Employee
NATURAL JOIN CHECKS
GROUP BY Employee_id,surname, name
ORDER BY Employee_id

SELECT Employee_id, surname, name, MIN(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY Employee_id,surname, name

SELECT Employee_id, surname, name, MAX(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY Employee_id,surname, name
ORDER BY Employee_id

SELECT Employee_id, surname, name, SUM(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY Employee_id,surname, name
ORDER BY Employee_id

SELECT VARIANCE(SALARY) 
FROM Employee
NATURAL JOIN Salary_change

SELECT STDDEV(SALARY) 
FROM Employee
NATURAL JOIN Salary_change

/*•	SQL S04 L03
o	COUNT, COUNT(DISTINCT ), NVL
o	Rozdíl mezi COUNT (*) a COUNT (atribut)
o	Proè NVL u agregaèních funkcí */


SELECT COUNT(employee_id) AS NO_EMPLOYEES
FROM employee

SELECT COUNT(*) 
FROM employee
WHERE Boemployment < TO_DATE('01/01/2018', 'DD/MM/YYYY')

SELECT count(DISTINCT(employee_id)) AS NO_EMPLOYEES
FROM employee


/* •	SQL S05 L01
o	GROUP BY
o	HAVING */

SELECT Employee_id, surname, name, AVG(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY Employee_id,surname, name
HAVING AVG(SALARY) > 100000
ORDER BY Employee_id


/* •	SQL S05 L02
o	ROLLUP, CUBE, ROUPING SETS */

SELECT Employee_ID, jobdesc, SUM(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY ROLLUP (Employee_ID, jobdesc)

SELECT Employee_ID, jobdesc, SUM(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY CUBE (Employee_ID, jobdesc)

SELECT Employee_ID, jobdesc, manager_id, SUM(SALARY)
FROM Employee
NATURAL JOIN Salary_change
GROUP BY GROUPING SETS 
((jobdesc, manager_id),(Employee_ID, jobdesc),
(Employee_ID, manager_id))

/*•	SQL S05 L03
o	Množinové operace v SQL – UNION, UNION ALL, INTERSECT, MINUS
o	ORDER BY u množinových operací*/

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE BoEmployment < TO_DATE('01/01/2018', 'DD/MM/YYYY')

UNION

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE Manager_ID IS NULL


SELECT  Employee_id, Surname, Name
FROM Employee
WHERE BoEmployment < TO_DATE('01/01/2018', 'DD/MM/YYYY')

UNION ALL

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE Manager_ID IS NULL

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE BoEmployment < TO_DATE('01/01/2018', 'DD/MM/YYYY')

INTERSECT

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE Manager_ID IS NULL


SELECT  Employee_id, Surname, Name
FROM Employee
WHERE BoEmployment < TO_DATE('01/01/2018', 'DD/MM/YYYY')

MINUS

SELECT  Employee_id, Surname, Name
FROM Employee
WHERE Manager_ID IS NULL

/*o	Vnoøené dotazy
o	Výsledek jako jediná hodnota
o	Vícesloupcový poddotaz
o	EXISTS, NOT EXISTS */

SELECT Employee_ID
FROM employee
NATURAL JOIN ADDRESS 
WHERE CITY_id = 
    (SELECT city_id
    FROM employee
    NATURAL JOIN ADDRESS 
    WHERE surname = 'Kraninger');

SELECT AVG(SALARY)
FROM Salary_change
NATURAL JOIN Employee
WHERE SALARY = 
    (SELECT salary
    FROM Salary_change
    NATURAL JOIN Employee
    WHERE surname = 'Kraninger');

SELECT Employee_id, MIN(salary)
FROM employee
NATURAL JOIN Salary_change
GROUP BY Employee_id
HAVING MIN(salary) >
(SELECT MIN(salary)
FROM employee
NATURAL JOIN Salary_change);

SELECT Employee_id, surname, name AS "Not a Manager"
FROM   employee emp 
WHERE NOT EXISTS 
(SELECT * 
FROM employee mgr
WHERE  mgr.manager_id = emp.employee_id)

SELECT Employee_id, surname, name AS "MANAGERS!"
FROM   employee emp 
WHERE EXISTS 
(SELECT * 
FROM employee mgr
WHERE  mgr.manager_id = emp.employee_id)


/*•	SQL S06 L03
o	Víceøádkové poddotazy IN, ANY, ALL
o	NULL hodnoty v poddotazech*/

SELECT Employee_id,Surname,name, boEmployment
FROM employee
WHERE EXTRACT(YEAR FROM boEmployment) IN
(SELECT EXTRACT(YEAR FROM boEmployment)
FROM employee
NATURAL JOIN address
WHERE city_id=1)
order by Employee_id

SELECT Employee_id,Surname,name, boEmployment
FROM employee
WHERE EXTRACT(YEAR FROM boEmployment) < ANY
(SELECT EXTRACT(YEAR FROM boEmployment)
FROM employee
NATURAL JOIN address
WHERE city_id=1)
ORDER BY Employee_id

SELECT Employee_id,Surname,name, boEmployment
FROM employee
WHERE EXTRACT(YEAR FROM boEmployment) < ALL
(SELECT EXTRACT(YEAR FROM boEmployment)
FROM employee
NATURAL JOIN address
WHERE city_id=1)
ORDER BY Employee_id

SELECT Surname, employee_id
FROM employee
WHERE employee_id IN
(SELECT manager_id
FROM employee);

SELECT  employee_id, manager_id, jobdesc
FROM    employee
WHERE  employee_id IN 
(SELECT  employee_id
FROM    employee
WHERE   surname = 'Kacalova' or surname = 'Kraninger')
AND    employee_id IN 
(SELECT employee_id 
FROM    employee
WHERE   EXTRACT(YEAR FROM boEmployment) IN                                        
2016)
AND employee_id NOT IN(1,4);

/*•	SQL S06 L04
o	WITH .. AS() konstrukce poddotazu*/

WITH managers AS
(SELECT DISTINCT manager_id
FROM employee
WHERE manager_id IS NOT NULL)

SELECT surname AS "Not a manager"
FROM employee
WHERE employee_id NOT IN 
(SELECT * 
FROM managers);

/*•	SQL S07 L01
o	INSERT INTO Tab VALUES()
o	INSERT INTo Tab (atr, atr) VALUES()
o	INSERT INTO Tab AS SELECT … */

INSERT INTO employee (employee_id,name,surname,phone,email,dob,bn,jobdesc,"User",login,boemployment,eoemployment,Record_date,address_id,manager_id)
VALUES (6,'Jana','Machová','+420605311565','Jana.Machova@hilite.com',TO_DATE('06/03/1990', 'DD/MM/YYYY'),'900306/5321','Accountant',1,
'MAC211',TO_DATE('03/03/2018', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,5,1);

INSERT INTO employee 
VALUES (7,'Margita','Malá','+420732693693','Margita.Mala@msa.com',TO_DATE('06/11/1976', 'DD/MM/YYYY'),'761106/5020','HR generalist',1,
'MAL001',TO_DATE('02/05/2015', 'DD/MM/YYYY'),TO_DATE(NULL),CURRENT_DATE,1,1);


CREATE TABLE sales_reps (
    id          INTEGER NOT NULL,
    surname     VARCHAR2(20) NOT NULL,
    name        VARCHAR2(20) NOT NULL,
    jobdesc     VARCHAR2(20) NOT NULL
);


INSERT INTO sales_reps(id, surname, name, jobdesc)   
SELECT employee_id, surname, name, jobdesc
FROM   employee
WHERE  jobdesc LIKE '%HR%';


/*•	SQL S07 L02
o	UPDATE Tab SET atr= …. WHERE podm
o	DELETE FROM Tab WHERE atr=… */

UPDATE Employee
SET phone = '+420603306090'
WHERE employee_id = 5;

DELETE FROM Employee
WHERE employee_id = 7;

/*•	SQL S07 L03
o	DEFAULT, MERGE, Multi-Table Inserts*/

CREATE TABLE my_employees (
    hire_date DATE DEFAULT SYSDATE,
    first_name VARCHAR2(15), 
    last_name VARCHAR2(15))

INSERT INTO employee 
VALUES (7,'Margita','Malá','+420732693693','Margita.Mala@msa.com',TO_DATE('06/11/1976', 'DD/MM/YYYY'),'761106/5020','HR generalist',1,
'MAL001',TO_DATE('02/05/2015', 'DD/MM/YYYY'),DEFAULT,CURRENT_DATE,1,1);


MERGE INTO sales_reps s  USING employee e
ON (s.id = e.employee_id)
WHEN MATCHED THEN UPDATE
SET
    s.surname = e.surname,
    s.jobdesc = e.jobdesc
WHEN NOT MATCHED THEN INSERT 
VALUES   (e.employee_id, e.surname, e.name, e.jobdesc)

INSERT ALL
    INTO sales_reps
    (id, surname, name, jobdesc)
    VALUES
    (304,'James','Parrot', 'IT_PROG')
    INTO sales_reps
    (id, surname, name, jobdesc)
    VALUES
    (305,'Rebecca','Graham','IT_PROG')
SELECT * FROM dual

/*o	Objekty v databází – Tabulky, Indexy, Constraint, View, Sequnce, Synonym
o	CREATE, ALTER, DROP, RENAME, TRUNCATE
o	CREATE TABLE (atr DAT TYP, DEFAULT NOT NULL )
o	ORGANIZATION EXTERNAL, TYPE ORACLE_LOADER, DEFAULT DICTIONARY, ACESS PARAMETERS, RECORDS DELIMITED BY NEWLINE, FIELDS, LOCATION */

CREATE TABLE my_friends
    (first_name VARCHAR2(20),
    last_name VARCHAR2(30),
    email VARCHAR2(30),
    phone_num VARCHAR2(12),
    birth_date DATE);

ALTER TABLE my_friends
ADD (release_date DATE DEFAULT SYSDATE);

ALTER TABLE my_friends
DROP COLUMN birth_date;

RENAME my_friends TO my_foes;

TRUNCATE TABLE my_foes;


CREATE TABLE Pokus(
    exact_time TIMESTAMP NOT NULL,
    hire_date DATE DEFAULT TO_DATE('02/05/2015', 'DD/MM/YYYY'),
    birthdate DATE DEFAULT SYSDATE
    );

/*                              NEJSEM SCHOPNY VYTVORIT OBJEKT NA SERVERU
CREATE TABLE Example(
    last_name CHAR(50),
    first_name CHAR(50),
    d CHAR(10)
    )
ORGANIZATION EXTERNAL
    (TYPE ORACLE_LOADER                                 
    DEFAULT DIRECTORY def_dir1 
    ACCESS PARAMETERS 
    (RECORDS DELIMITED BY NEWLINE 
    FIELDS (
        last_name CHAR(50),
        first_name CHAR(50), 
        d   CHAR(10) date_format DATE mask "mm/dd/yyyy"
        )
    )
LOCATION ('info.dat'));
*/

/*
•	SQL S08 L02
o	TIMESTAMP, TIMESTAMP WITH TIME ZONE, TIMESTAM WITH LOCAL TIMEZONE
o	INTERVAL YEAT TO MONTH, INTERVAL DAY TO SECOND
o	CHAR, VARCHAR2, CLOB
o	NUMBER
o	BLOB
*/

CREATE TABLE T(
    exact_time TIMESTAMP,
    t1 TIMESTAMP WITH TIME ZONE,
    t2 TIMESTAMP WITH LOCAL TIME ZONE
    );

INSERT INTO T (exact_time,t1,t2)
VALUES (SYSDATE,'06.11.21 15:37:02,000000000 EUROPE/PRAGUE','06.11.21 15:37:02')

CREATE TABLE time_ex4
    (delka1 INTERVAL YEAR(3) TO MONTH,
     delka2 INTERVAL YEAR(2) TO MONTH);

INSERT INTO time_ex4 (delka1, delka2)
VALUES (INTERVAL '120' MONTH(3), 
        INTERVAL '3-6' YEAR TO MONTH);

SELECT  SYSDATE + delka1 AS "120 months from now", 
        SYSDATE + delka2 AS "3 years 6 months from now"
FROM time_ex4;

CREATE TABLE ccc(   
        c CHAR(10),
        cc VARCHAR (50),
        ccc CLOB,
        n   number,
        b   BLOB
        )
        
INSERT INTO ccc (c, cc, ccc,n,b)
VALUES ('Michal','Kraninger', 'Pokus', 125, utl_raw.cast_to_raw('fuck you!!!'));    

/*
•	SQL S08 L03
o	ALTER TABLE (ADD, MODIFY, DROP), DROP, RENAME
o	FLASHBACK TABLE Tab TO BEFORE DROP (pohled USER_RECYCLEBIN)
o	DELETE, TRUNCATE
o	COMMENT ON TABLE
o	SET UNUSED */

ALTER TABLE my_friends
ADD (hobbies  VARCHAR(50));

ALTER TABLE my_friends
MODIFY (hobbies VARCHAR2(20));

DROP TABLE my_friends;

RENAME my_friends to my_foes;

DROP TABLE ccc;

FLASHBACK TABLE ccc TO BEFORE DROP;

SHOW RECYCLEBIN;

DELETE FROM ccc;

TRUNCATE TABLE CCC;

COMMENT ON TABLE employee
IS 'all employees of the company';

SELECT table_name, comments 
FROM user_tab_comments;

ALTER TABLE ccc
SET UNUSED (ccc);

CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    CONSTRAINT UC_Person UNIQUE (ID,LastName)
); 


CREATE TABLE sales_reps (
    id          INTEGER NOT NULL,
    surname     VARCHAR2(20) NOT NULL,
    name        VARCHAR2(20) NOT NULL,
    jobdesc     VARCHAR2(20) NOT NULL
);

INSERT INTO sales_reps(id, surname, name, jobdesc)   
SELECT employee_id, surname, name, jobdesc
FROM   employee
WHERE  jobdesc LIKE '%HR%';

/*•	SQL S10 L02
o	CONSTRAINT – NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY (atr REFERENCES Tab(atr) ), CHECK
o	Cizí klíèe, ON DELETE, ON UPDATE, RESTRICT, CASCADE atd.*/


CREATE TABLE K(
    a     INTEGER CONSTRAINT PK PRIMARY KEY,
    b     VARCHAR2(20) NOT NULL,
    c     VARCHAR2(20) CONSTRAINT c_u UNIQUE,
    d     INTEGER,
    e     date CONSTRAINT date_ck CHECK (e> TO_DATE('02/05/2015', 'DD/MM/YYYY')),
   
CONSTRAINT Employee_id_fk FOREIGN KEY (d) REFERENCES employee(employee_id) ON DELETE SET NULL);

INSERT INTO K (a, b, c, d, e)
VALUES (1,'Michal','Kraninger',1,TO_DATE('03/05/2015', 'DD/MM/YYYY'));   
INSERT INTO K (a, b, c, d, e)
VALUES (2,'Petr','Gajda',10,TO_DATE('05/05/2015', 'DD/MM/YYYY'));   

CREATE TABLE K2(
    a     INTEGER CONSTRAINT PK1 PRIMARY KEY,
    b     VARCHAR2(20) NOT NULL,
    c     VARCHAR2(50), 
    d     INTEGER,
CONSTRAINT Employee_id_fk_1 FOREIGN KEY (d) REFERENCES employee(employee_id) ON DELETE CASCADE);

ALTER TABLE K2
DROP CONSTRAINT Employee_id_fk_1

/*
•	SQL S10 L03
o	USER_CONSTRAINTS */

SELECT constraint_name, table_name, constraint_type, status
FROM USER_CONSTRAINTS
WHERE table_name ='Employee'

/*•	SQL S11 L01
•	SQL S11 L01
o	CREATE VIEW
o	FORCE, NOFORCE
o	WITCH CHECK OPTION
o	WITH READ ONLY */


CREATE VIEW view_employees
AS SELECT employee_id,name, email 
FROM employee
WHERE employee_id BETWEEN 1 and 5;

CREATE FORCE VIEW Pavel
AS SELECT *
FROM wf_countries;

CREATE NOFORCE VIEW Pavlina
AS SELECT *
FROM wf_countries;

CREATE VIEW view_employees
AS SELECT employee_id,name, email 
FROM employee
WHERE employee_id BETWEEN 1 and 5;

CREATE VIEW Salary
AS 
SELECT employee_id,salary,date_modified
FROM salary_change
WITH READ ONLY; 

CREATE VIEW Salary1
AS 
SELECT employee_id,salary,date_modified
FROM salary_change
WITH CHECK OPTION;

CREATE OR REPLACE VIEW employees
AS SELECT employee_id, surname, name, phone,email
FROM employee
WHERE surname LIKE '%K%';

CREATE OR REPLACE VIEW employees  ("ID", "Surname", "First name", "Salary")
AS SELECT e.employee_id, e.surname, e.name, s.salary
FROM employee e JOIN Salary_change s
ON e.employee_id = s.employee_id
WHERE surname LIKE '%K%';

/*
•	SQL S11 L03
o	INLINE VIEW Poddotaz v podobì tabulky SELECT atr FROM (SELECT * FROM Tab) alt_tab */

SELECT e.surname, e.name, e.phone, a.maxaddress
FROM employee e,
    (SELECT address_id, max(address_id) maxaddress
    FROM address
    GROUP BY address_id) a
WHERE e.address_id = a.address_id
AND e.address_id = a.maxaddress;

/*
•	SQL S12 L01
o	CREATE SEQUENCE nazev INCREMENT BY n START WITH m, (NO)MAXVALUE, (NO)MINVALUE, 
    ALTER SEQUENCE
*/

CREATE SEQUENCE Sekvence2
    INCREMENT BY 1
    START WITH 1
    MAXVALUE 50000
    MINVALUE 1
    NOCACHE
    NOCYCLE;

ALTER SEQUENCE Sekvence2
    INCREMENT BY 1
    MAXVALUE 999999
    NOCACHE
    NOCYCLE;
        
SELECT sequence_name, min_value, max_value, increment_by, 
last_number
FROM user_sequences;

/*
    •	SQL S12 L02
    o	CREATE INDEX, PRIMARY KEY, UNIQUE KEY, FOREINGN KEY
*/

CREATE INDEX check_date_index
ON checks(Check_date);

/*
•	SQL S13 L01
o	GRANT … ON … TO … PUBLIC
*/

GRANT SELECT ON employee TO public;

REVOKE SELECT ON employee FROM public;

/*
o	Jaká práva lze pøidìlit na jaké objekty? (ALTER, DELETE, EXECUTE, INDEX, INSERT, REFERENCES, SELECT, UPDATE) – (TABLE, VIEW, SEQUENCE, PROCEDURE)
•	SQL S13 L03
o	Regulární výrazy
o	REGEXP_LIKE, REGEXP_REPLACE, REGEXP_INSTR, REGEXP_SUBSTR, REGEXP_COUNT */

GRANT ALTER, DELETE, INDEX, INSERT, REFERENCES, SELECT, UPDATE ON Whours
TO PUBLIC;

GRANT select,delete, update, INSERT, REFERENCES ON Employees
TO PUBLIC;

GRANT ALTER, SELECT ON Sekvence2
TO PUBLIC;

CREATE OR REPLACE PROCEDURE procedure_1
   
IS
    /*int a;*/

BEGIN
    DBMS_OUTPUT.PUT_LINE('Tisk procedury procedure_1');
END;

GRANT EXECUTE ON procedure_1
TO PUBLIC;

/*

•	SQL S13 L03
o	Regulární výrazy
o	REGEXP_LIKE, REGEXP_REPLACE, REGEXP_INSTR, REGEXP_SUBSTR, REGEXP_COUNT

*/

SELECT surname, name
FROM employee
WHERE REGEXP_LIKE(name, '^Mich(a|ae)l$');

SELECT surname, REGEXP_REPLACE(surname, '^K(a|e|i|o|u)', 
'**')
AS "Name changed"
FROM employee;

SELECT surname, REGEXP_COUNT(surname, '(K)') AS 
"Count of 'K'"
FROM employee
WHERE REGEXP_COUNT(surname, '(K)')>0;

SELECT employee_id,surname, REGEXP_INSTR(surname, 'a|e|i|o|u', 1, 1, 0, 'i') AS first_occurrence
FROM employee;

SELECT employee_id,surname, REGEXP_SUBSTR (surname, 'a|e|i|o|u', 1, 1, 'i') AS "First Vowel"
FROM employee;

/*
•	SQL S14 L01
o	Transakce, COMMIT, ROLLBACK, SAVEPOINT
*/

INSERT INTO country(country_id,name)
VALUES(6,'The US');

COMMIT WORK;

UPDATE factor
    SET factor_name = 'Level of noise'
    WHERE factor_id = 1; 
SAVEPOINT one;

Rollback to SAVEPOINT one;

/*
•	SQL S15 L01
o	Alternativní zápis spojování bez JOIN s podmínkou spojení ve WHERE
o	Levé a pravé spojení s pomocí atrA = atrB (+)
*/

SELECT employee.employee_id, employee.surname, employee.name, salary_change.salary
FROM  Employee, salary_change 
WHERE Employee.employee_id = salary_change.employee_id;

SELECT e.employee_id, e.surname, e.name, s.salary
FROM  Employee e, salary_change s 
WHERE e.employee_id (+) = s.employee_id;

SELECT e.employee_id, e.surname, e.name, s.salary
FROM  Employee e, salary_change s 
WHERE e.employee_id = s.employee_id(+);







