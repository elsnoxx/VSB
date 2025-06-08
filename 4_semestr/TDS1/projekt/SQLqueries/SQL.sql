-- DD S15 L01
-- Write query for concatenate strings by pipes || , and CONCAT() SELECT DISTINCT 

SELECT firstname || '|' || lastname AS full_name
FROM Guest;

SELECT CONCAT(firstname, '|') || lastname AS full_name
FROM Guest;

SELECT DISTINCT firstname || ' ' || lastname AS full_name
FROM Guest;

-- DD S16 L02
-- WHERE condition for selecting rows Functions LOWER, UPPER, INITCAP
SELECT firstname, lastname,
       LOWER(firstname) AS lower_firstname,
       UPPER(lastname) AS upper_lastname,
       INITCAP(firstname || ' ' || lastname) AS initcap_fullname
FROM Guest
WHERE city = 'Prague';

-- DD S16 L03
-- BETWEEN … AND LIKE (%, _) IN(), IS NULL, IS NOT NULL


-- DD S17 L01
-- AND, OR, NOT, Evaluation priority ()

SELECT firstname, lastname, city
FROM Guest
WHERE (city = 'Prague' OR city = 'Brno')
  AND (country = 'Czech Republic' OR country = 'Slovakia')
  AND NOT (postal_code IS NULL)
    AND (firstname LIKE 'A%' OR lastname LIKE '%ova')
ORDER BY city, lastname;


-- DD S17 L02
-- ORDER BY atr [ASC/DESC], Sorting by using one or more attributes
SELECT firstname, lastname, city, postal_code
FROM Guest
WHERE city = 'Prague'
ORDER BY city ASC, lastname DESC;



-- DD S17 L03
-- Single row functions, Column functions MIN, MAX, AVG, SUM, COUNT
SELECT 
    MIN(accommodation_price) AS min_price,
    MAX(accommodation_price) AS max_price,
    AVG(accommodation_price) AS avg_price,
    SUM(accommodation_price) AS total_price,
    COUNT(*) AS total_reservations
FROM Reservation
WHERE status = 'confirmed';

-- SQL S01 L01
-- LOWER, UPPER, INITCAP
-- CONCAT, SUBSTR, LENGTH, INSTR, LPAD, RPAD, TRIM, REPLACE
-- Use virtual table DUAL

SELECT
    LOWER('HELLO WORLD') AS lower_case,       
    UPPER('hello world') AS upper_case,       
    INITCAP('hello world') AS initcap_case,   
    CONCAT('Hello', ' World') AS concatenated,
    SUBSTR('Hello World', 1, 5) AS substring, 
    LENGTH('Hello World') AS length_of_string,
    INSTR('Hello World', 'World') AS position,
    LPAD('123', 5, '0') AS left_padded,       
    RPAD('123', 5, '0') AS right_padded,      
    TRIM('  Hello World  ') AS trimmed,       
    REPLACE('Hello World', 'World', 'Oracle') AS replaced
FROM DUAL;

-- • SQL S01 L02
-- ROUND, TRUNC round for two decimal places, whole thousands MOD

SELECT 
    total_accommodation,
    total_expenses,
    ROUND(total_accommodation * 1.1, 2) AS rounded_total_with_tax,
    TRUNC(total_accommodation, -3) AS truncated_to_thousands,
    MOD(total_accommodation, 1000) AS remainder_thousands
FROM Payment;

-- SQL S01 L03
-- MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY, ROUND, TRUNC, System constant SYSDATE
SELECT 
    payment_date,
    MONTHS_BETWEEN(SYSDATE, payment_date) AS months_between_now_and_payment,
    ADD_MONTHS(payment_date, 3) AS payment_plus_3_months,
    NEXT_DAY(payment_date, 'MONDAY') AS next_monday_after_payment,
    LAST_DAY(payment_date) AS last_day_of_payment_month,
    ROUND(MONTHS_BETWEEN(SYSDATE, payment_date)) AS rounded_months_between,
    TRUNC(payment_date, 'MONTH') AS first_day_of_payment_month
FROM Payment;

-- SQL S02 L01
-- o TO_CHAR, TO_NUMBER, TO_DATE

SELECT 
    TO_CHAR(payment_date, 'DD-MON-YYYY') AS formatted_date,
    TO_NUMBER('12345.67', '99999.99') AS converted_number,
    TO_DATE('2023-10-01', 'YYYY-MM-DD') AS converted_date
FROM Payment;


-- SQL S02 L02
-- o NVL, NVL2, NULLIF, COALESCE
SELECT 
    NVL(accommodation_price, 0) AS accommodation_price_or_zero,
    NVL2(accommodation_price, 'Price available', 'No price') AS price_status,
    NULLIF(accommodation_price, 0) AS price_if_not_zero,
    COALESCE(accommodation_price, total_expenses, 0) AS first_non_null_value
FROM Reservation;

-- • SQL S02 L03
-- DECODE, CASE, IF-THEN-ELSE

SELECT 
    DECODE(status, 
           'confirmed', 'Reservation confirmed', 
           'cancelled', 'Reservation cancelled', 
           'pending', 'Reservation pending', 
           'Unknown status') AS reservation_status,
    CASE 
        WHEN accommodation_price > 1000 THEN 'High price'
        WHEN accommodation_price BETWEEN 500 AND 1000 THEN 'Medium price'
        ELSE 'Low price'
    END AS price_category
FROM Reservation;

-- • SQL S03 L01
-- NATURAL JOIN, CROSS JOIN

SELECT 
    e.firstname || ' ' || e.lastname AS employee_name,
    d.department_name
FROM Employee e
CROSS JOIN Department d;

-- • SQL S03 L02
-- JOIN … USING(atr), JOIN .. ON (joining condition)

SELECT 
    e.firstname || ' ' || e.lastname AS employee_name,
    d.department_name
FROM Employee e
JOIN Department d ON e.department_id = d.department_id;

-- • SQL S03 L03
-- o LEFT OUTER JOIN … ON ()
-- o RIGHT OUTER JOIN … ON ()
-- o FULL OUTER JOIN … ON ()
-- • SQL S03 L04
-- o Joining 2x of the same table with renaming (link between superiors and subordinates
-- in one table)
-- o Hierarchical querying – tree structure of START WITH, CONNECT BY PRIOR, LEVEL
-- dive
-- • SQL S04 L02
-- o AVG, COUNT, MIN, MAX, SUM, VARIANCE, STDDEV
-- • SQL S04 L03
-- o COUNT, COUNT(DISTINCT ), NVL
-- o Difference between COUNT (*) a COUNT (attribute)
-- o Why using NVL for aggregation functions
-- • SQL S05 L01
-- o GROUP BY
-- o HAVING
-- • SQL S05 L02
-- o ROLLUP, CUBE, ROUPING SETS
-- • SQL S05 L03
-- o Multiple operations in SQL – UNION, UNION ALL, INTERSECT, MINUS
-- o ORDER BY for set operations
-- • SQL S06 L01
-- o Nested queries
-- o Result as a single value
-- o Multi-column subquery
-- o EXISTS, NOT EXISTS
-- • SQL S06 L02
-- o One-line subqueries
-- • SQL S06 L03
-- o Multi-line subqueries IN, ANY, ALL
-- o NULL values in subqueries
-- • SQL S06 L04
-- o WITH .. AS() subquery construction
-- • SQL S07 L01
-- o INSERT INTO Tab VALUES()
-- o INSERT INTo Tab (atr, atr) VALUES()
-- o INSERT INTO Tab AS SELECT …
-- SQL S07 L02
-- o UPDATE Tab SET atr= …. WHERE condition
-- o DELETE FROM Tab WHERE atr=…
-- • SQL S07 L03
-- o DEFAULT, MERGE, Multi-Table Inserts
-- • SQL S08 L01
-- o Objects in databases – Tables, Indexes, Constraint, View, Sequence, Synonym
-- o CREATE, ALTER, DROP, RENAME, TRUNCATE
-- o CREATE TABLE (atr DAT TYPE, DEFAULT NOT NULL )
-- o ORGANIZATION EXTERNAL, TYPE ORACLE_LOADER, DEFAULT DICTIONARY, ACCESS
-- PARAMETERS, RECORDS DELIMITED BY NEWLINE, FIELDS, LOCATION
-- • SQL S08 L02
-- o TIMESTAMP, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH LOCAL TIMEZONE
-- o INTERVAL YEAT TO MONTH, INTERVAL DAY TO SECOND
-- o CHAR, VARCHAR2, CLOB
-- o about NUMBER
-- o about BLOB
-- • SQL S08 L03
-- o ALTER TABLE (ADD, MODIFY, DROP), DROP, RENAME
-- o FLASHBACK TABLE Tab TO BEFORE DROP (view USER_RECYCLEBIN)
-- o DELETE, TRUNCATE
-- o COMMENT ON TABLE
-- o SET UNUSED
-- • SQL S10 L01
-- o CREATE TABLE (NOT NULL AND UNIQUE constraint)
-- o CREATE TABLE Tab AS SELECT …
-- o Own vs. system naming CONSTRAINT conditions
-- • SQL S10 L02
-- o CONSTRAINT – NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY (atr REFERENCES
-- Tab(atr) ), CHECK
-- o Foreign keys, ON DELETE, ON UPDATE, RESTRICT, CASCADE, etc.
-- • SQL S10 L03
-- o about USER_CONSTRAINTS
-- • SQL S11 L01
-- o CREATE VIEW
-- o about FORCE, NOFORCE
-- o WITCH CHECK OPTION
-- o WITH READ ONLY
-- o about Simple vs. Compex VIEW
-- • SQL S11 L03
-- o INLINE VIEW Subquery in the form of a table SELECT atr FROM (SELECT * FROM Tab)
-- alt_tab
-- • SQL S12 L01
-- o CREATE SEQUENCE name INCREMENT BY n START WITH m, (NO)MAXVALUE,
-- (NO)MINVALUE, (NO)CYCLE, (NO)CACHE
-- o about ALTER SEQUENCE
-- • SQL S12 L02
-- o CREATE INDEX, PRIMARY KEY, UNIQUE KEY, FOREIGN KEY
-- • SQL S13 L01
-- o GRANT … ON … TO … PUBLIC
-- o about REVOKE
-- o What rights can be assigned to which objects? (ALTER, DELETE, EXECUTE, INDEX,
-- INSERT, REFERENCES, SELECT, UPDATE) – (TABLE, VIEW, SEQUENCE, PROCEDURE)
-- • SQL S13 L03
-- o Regular expressions
-- o REGEXP_LIKE, REGEXP_REPLACE, REGEXP_INSTR, REGEXP_SUBSTR, REGEXP_COUNT
-- • SQL S14 L01
-- o Transactions, COMMIT, ROLLBACK, SAVEPOINT
-- • SQL S15 L01
-- o Alternative join notation without JOIN with join condition in WHERE
-- o Left and right connection using atrA = atrB (+)
-- • SQL S16 L03
-- o Recapitulation of commands and parameters - complete everything that was not
-- mentioned in the previous points here