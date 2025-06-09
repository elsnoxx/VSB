-- DD S15 L01
-- Write query for concatenate strings by pipes || , and CONCAT() SELECT DISTINCT 

SELECT firstname || '|' || lastname AS full_name
FROM Guest;

SELECT CONCAT(firstname, '|') || lastname AS full_name
FROM Guest;

SELECT DISTINCT firstname || ' ' || lastname AS full_name
FROM Guest;
/*********************************************************************************************************/
-- DD S16 L02
-- WHERE condition for selecting rows Functions LOWER, UPPER, INITCAP
SELECT
  employee_id, firstname, lastname, email
FROM
  employee
WHERE
  UPPER(lastname) = 'Svoboda'; 
    
SELECT
  guest_id, firstname, lastname, email
FROM
  Guest
WHERE
  lower(email) = 'ellen.kacalova@msa.com'; 

SELECT INITCAP(email)
FROM Guest;

/*********************************************************************************************************/
-- DD S16 L03
-- BETWEEN … AND LIKE (%, _) IN(), IS NULL, IS NOT NULL

SELECT * FROM Payment
where total_accommodation BETWEEN 2000 AND 2500;


SELECT *
FROM EMPLOYEE 
WHERE SURNAME LIKE 'K%';

SELECT *
FROM EMPLOYEE 
WHERE Employee_id IN (select employee_id from EMPLOYEE);

SELECT *
FROM feedback 
WHERE note IS NULL;
/*********************************************************************************************************/
-- DD S17 L01
-- AND, OR, NOT, Evaluation priority ()

-- Vyber hosty z Prahy nebo Brna, kteří jsou z České republiky nebo Slovenska
SELECT firstname, lastname, city
FROM Guest
WHERE (city = 'Prague' OR city = 'Brno')
  AND (country = 'ČR' OR country = 'SK');

-- Vyber hosty, kteří nemají vyplněný PSČ
SELECT firstname, lastname, postal_code
FROM Guest
WHERE NOT (postal_code IS NOT NULL);

-- Vyber hosty, jejichž jméno začíná na "A" nebo jejich příjmení končí na "ova"
SELECT firstname, lastname
FROM Guest
WHERE (firstname LIKE 'A%' OR lastname LIKE '%ova');

-- Kombinace podmínek s prioritizací závorek
SELECT firstname, lastname, city
FROM Guest
WHERE (city = 'Prague' OR city = 'Znojmo')
  AND NOT (postal_code IS NULL)
  AND (firstname LIKE 'A%' OR lastname LIKE '%ova');

/*********************************************************************************************************/
-- DD S17 L02
-- ORDER BY atr [ASC/DESC], Sorting by using one or more attributes
SELECT firstname, lastname, city, postal_code
FROM Guest
WHERE city = 'Brno'
ORDER BY city ASC, lastname DESC;


/*********************************************************************************************************/
-- DD S17 L03
-- Single row functions, Column functions MIN, MAX, AVG, SUM, COUNT
SELECT 
    MIN(accommodation_price) AS min_price,
    MAX(accommodation_price) AS max_price,
    AVG(accommodation_price) AS avg_price,
    SUM(accommodation_price) AS total_price,
    COUNT(*) AS total_reservations
FROM Reservation
WHERE status = 'Confirmed';
/*********************************************************************************************************/
-- SQL S01 L01
-- LOWER, UPPER, INITCAP
-- CONCAT, SUBSTR, LENGTH, INSTR, LPAD, RPAD, TRIM, REPLACE
-- Use virtual table DUAL

-- Ukázka použití funkcí na datech z tabulky Guest
SELECT
    LOWER(firstname) AS lower_case_firstname,
    UPPER(lastname) AS upper_case_lastname,
    INITCAP(city) AS initcap_city,
    CONCAT(firstname, ' ') || lastname AS full_name,
    SUBSTR(email, 1, INSTR(email, '@') - 1) AS email_username,
    LENGTH(email) AS email_length,
    INSTR(email, '@') AS at_position,
    LPAD(postal_code, 10, '0') AS left_padded_postal_code,
    RPAD(postal_code, 10, '0') AS right_padded_postal_code,
    TRIM(' ' FROM notes) AS trimmed_notes,
    REPLACE(country, 'ČR', 'Czech Republic') AS replaced_country
FROM Guest
WHERE city = 'Praha' OR city = 'Brno';
/*********************************************************************************************************/
-- • SQL S01 L02
-- ROUND, TRUNC round for two decimal places, whole thousands MOD

SELECT 
    total_accommodation,
    total_expenses,
    ROUND(total_accommodation * 1.1, 2) AS rounded_total_with_tax,
    TRUNC(total_accommodation, -3) AS truncated_to_thousands,
    MOD(total_accommodation, 1000) AS remainder_thousands
FROM Payment;
/*********************************************************************************************************/
-- SQL S01 L03
-- MONTHS_BETWEEN, ADD_MONTHS, NEXT_DAY, LAST_DAY, ROUND, TRUNC, System constant SYSDATE
SELECT TO_CHAR(MONTHS_BETWEEN 
   (TO_DATE('09-06-2022','MM-DD-YYYY'),
    SYSDATE))
    FROM DUAL;

SELECT TO_DATE(
    'January 15, 1989, 11:00 A.M.',
    'Month dd, YYYY, HH:MI A.M.',
     'NLS_DATE_LANGUAGE = American')
     FROM DUAL;
     
SELECT
TO_NUMBER('4687841', '9999999')
FROM DUAL;
/*********************************************************************************************************/
-- SQL S02 L01
-- o TO_CHAR, TO_NUMBER, TO_DATE

SELECT 
    TO_CHAR(payment_date, 'DD-MON-YYYY') AS formatted_date,
    TO_NUMBER('12345.67', '99999.99') AS converted_number,
    TO_DATE('2023-10-01', 'YYYY-MM-DD') AS converted_date
FROM Payment;

/*********************************************************************************************************/
-- SQL S02 L02
-- o NVL, NVL2, NULLIF, COALESCE
SELECT 
    NVL(total_accommodation, 0) AS accommodation_price_or_zero,
    NVL2(total_accommodation, 'Price available', 'No price') AS price_status,
    NULLIF(total_accommodation, 0) AS price_if_not_zero,
    COALESCE(total_accommodation, total_expenses, 0) AS first_non_null_value
FROM Payment;
/*********************************************************************************************************/
-- • SQL S02 L03
-- DECODE, CASE, IF-THEN-ELSE

SELECT 
    DECODE(status, 
           'Confirmed', 'Reservation confirmed', 
           'Cancelled', 'Reservation cancelled', 
           'Pending', 'Reservation pending', 
           'Unknown status') AS reservation_status,
    CASE 
        WHEN accommodation_price > 1000 THEN 'High price'
        WHEN accommodation_price BETWEEN 500 AND 1000 THEN 'Medium price'
        ELSE 'Low price'
    END AS price_category
FROM Reservation;
/*********************************************************************************************************/
-- • SQL S03 L01
-- NATURAL JOIN, CROSS JOIN

-- natural join
SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date
FROM Guest g NATURAL JOIN Reservation r;

-- cross join
SELECT g.firstname, g.lastname, r.room_number
FROM Guest g CROSS JOIN Room r;

/*********************************************************************************************************/
-- • SQL S03 L02
-- JOIN … USING(atr), JOIN .. ON (joining condition)

-- Using JOIN ... USING(atr)
SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date
FROM Reservation r JOIN Guest g USING (guest_id);

-- Using JOIN ... ON (joining condition)
SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date, p.total_accommodation
FROM Guest g
JOIN Reservation r ON g.guest_id = r.guest_id
JOIN Payment p ON r.payment_id = p.payment_id
WHERE p.is_paid = 0;
/*********************************************************************************************************/
-- • SQL S03 L03
-- LEFT OUTER JOIN … ON ()
-- Find all guests and their reservations (including guests with no reservations)
SELECT g.guest_id, g.firstname, g.lastname, r.reservation_id, r.check_in_date, r.check_out_date
FROM Guest g LEFT OUTER JOIN Reservation r ON g.guest_id = r.guest_id
ORDER BY g.lastname;

-- RIGHT OUTER JOIN … ON ()
-- Find all room types and their rooms (including room types with no rooms)
SELECT r.room_id, r.room_number, rt.room_type_id, rt.name AS room_type_name 
FROM Room r RIGHT OUTER JOIN RoomType rt ON r.room_type_id = rt.room_type_id
ORDER BY rt.name;

-- FULL OUTER JOIN … ON ()
-- Match guests with their feedback, showing all guests and all feedback entries
SELECT g.guest_id, g.firstname, g.lastname, 
       f.feedback_id, f.rating, f.note, f.feedback_date
FROM Guest g
FULL OUTER JOIN Feedback f ON g.guest_id = f.guest_id
ORDER BY g.lastname, f.feedback_date;

/*********************************************************************************************************/
-- • SQL S03 L04
-- o Joining 2x of the same table with renaming (link between superiors and subordinates in one table)
-- Find pairs of employees working in the same city
SELECT e1.employee_id AS emp1_id,
       e1.firstname || ' ' || e1.lastname AS employee1_name,
       e2.employee_id AS emp2_id,
       e2.firstname || ' ' || e2.lastname AS employee2_name,
       e1.city
FROM Employee e1
JOIN Employee e2 ON e1.city = e2.city
WHERE e1.employee_id < e2.employee_id
ORDER BY e1.city, e1.employee_id;

-- o Hierarchical querying – tree structure of START WITH, CONNECT BY PRIOR, LEVEL
SELECT LEVEL, employee_id, 
       LPAD(' ', (LEVEL-1)*2) || firstname || ' ' || lastname AS employee_name, 
       position, manager_id
FROM Employee
START WITH manager_id IS NULL
CONNECT BY PRIOR employee_id = manager_id
ORDER SIBLINGS BY employee_id;

/*********************************************************************************************************/
-- • SQL S04 L02
-- o AVG, COUNT, MIN, MAX, SUM, VARIANCE, STDDEV

-- Average guest rating from feedback
SELECT AVG(rating) AS avg_rating 
FROM Feedback;

-- Count guests by guest type
SELECT guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY guest_type;

-- Lowest guest rating
SELECT MIN(rating) AS lowest_rating
FROM Feedback;

-- Maximum price of any room
SELECT MAX(price_per_night) AS most_expensive_room
FROM RoomTypePriceHistory
WHERE valid_to IS NULL;

-- Total revenue from all reservations
SELECT SUM(accommodation_price) AS total_accommodation_revenue
FROM Reservation;

-- Variance in guest ratings
SELECT VARIANCE(rating) AS rating_variance
FROM Feedback;

-- Standard deviation of guest ratings
SELECT STDDEV(rating) AS rating_stddev
FROM Feedback;

/*********************************************************************************************************/
-- • SQL S04 L03
-- o COUNT, COUNT(DISTINCT ), NVL

-- Count total number of guests
SELECT COUNT(*) AS total_guests 
FROM Guest;

-- Count distinct cities where guests come from
SELECT COUNT(DISTINCT city) AS unique_cities
FROM Guest;

-- Use NVL to replace NULL notes with 'No notes provided'
SELECT guest_id, firstname, lastname, NVL(notes, 'No notes provided') AS guest_notes
FROM Guest;

-- o Difference between COUNT (*) a COUNT (attribute)

-- Similarly for Payment table
SELECT 
    COUNT(*) AS total_payments,
    COUNT(payment_date) AS payments_with_date,
    COUNT(*) - COUNT(payment_date) AS payments_without_date
FROM Payment;

-- o Why using NVL for aggregation functions
-- Count guest notes with and without NVL
SELECT 
    COUNT(notes) AS count_without_nvl,
    COUNT(NVL(notes, 'None')) AS count_with_nvl
FROM Guest;
/*********************************************************************************************************/

-- • SQL S05 L01
-- o GROUP BY
-- Count guests by city
SELECT city, COUNT(*) AS guest_count
FROM Guest
GROUP BY city
ORDER BY guest_count DESC;

-- o HAVING
-- Find cities with more than 1 guest
SELECT city, COUNT(*) AS guest_count
FROM Guest
GROUP BY city
HAVING COUNT(*) > 1
ORDER BY guest_count DESC;

/*********************************************************************************************************/
-- • SQL S05 L02
-- o ROLLUP, CUBE, ROUPING SETS

-- Analyze guest distribution by city and guest_type with subtotals
SELECT city, guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY ROLLUP(city, guest_type)
ORDER BY city, guest_type;

-- Multi-dimensional analysis of guest counts by city and guest_type
SELECT city, guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY CUBE(city, guest_type)
ORDER BY city, guest_type;

/*********************************************************************************************************/
-- • SQL S05 L03
-- o Multiple operations in SQL – UNION, UNION ALL, INTERSECT, MINUS
-- Find all cities where either guests or employees are from
SELECT city, 'Guest' AS person_type
FROM Guest
UNION
SELECT city, 'Employee' AS person_type
FROM Employee
ORDER BY city;

-- List all services used by reservation status (keeping duplicates)
SELECT r.status, s.name AS service_name
FROM Reservation r
JOIN ServiceUsage su ON r.reservation_id = su.reservation_id
JOIN Service s ON su.service_id = s.service_id
WHERE r.status = 'Confirmed'
UNION ALL
SELECT r.status, s.name AS service_name
FROM Reservation r
JOIN ServiceUsage su ON r.reservation_id = su.reservation_id
JOIN Service s ON su.service_id = s.service_id
WHERE r.status = 'Pending'
ORDER BY status, service_name;

-- Find cities where both guests and employees live
SELECT city FROM Guest
INTERSECT
SELECT city FROM Employee
ORDER BY city;

-- Find cities where guests live but no employees
SELECT city FROM Guest
MINUS
SELECT city FROM Employee
ORDER BY city;

-- o ORDER BY for set operations
-- List all people associated with the hotel, sorted by type and name
SELECT firstname, lastname, 'Guest' AS person_type
FROM Guest
UNION
SELECT firstname, lastname, 'Employee' AS person_type
FROM Employee
ORDER BY person_type, lastname, firstname;

/*********************************************************************************************************/
-- • SQL S06 L01
-- o Nested queries
-- o Result as a single value
-- Find reservations with accommodation price above average
SELECT r.reservation_id, g.firstname, g.lastname, r.accommodation_price
FROM Reservation r
JOIN Guest g ON r.guest_id = g.guest_id
WHERE r.accommodation_price > (
    SELECT AVG(accommodation_price) 
    FROM Reservation
)
ORDER BY r.accommodation_price DESC;

-- o Multi-column subquery
-- Find guests who have the same city and guest_type as VIP guests from Prague
SELECT g.guest_id, g.firstname, g.lastname, g.city, g.guest_type
FROM Guest g
WHERE (g.city, g.guest_type) IN (
    SELECT city, guest_type
    FROM Guest
    WHERE guest_type = 'standard' AND city = 'Praha'
)
ORDER BY g.lastname;

-- o EXISTS, NOT EXISTS
-- Find guests who have at least one confirmed reservation
SELECT g.guest_id, g.firstname, g.lastname
FROM Guest g
WHERE EXISTS (
    SELECT 1
    FROM Reservation r
    WHERE r.guest_id = g.guest_id
    AND r.status = 'Confirmed'
)
ORDER BY g.lastname;


/*********************************************************************************************************/
-- • SQL S06 L02
-- o One-line subqueries
-- Calculate how much each reservation differs from the average accommodation price
SELECT r.reservation_id, r.guest_id, r.accommodation_price, 
       r.accommodation_price - (SELECT AVG(accommodation_price) FROM Reservation) AS price_difference
FROM Reservation r
ORDER BY price_difference DESC;

/*********************************************************************************************************/
-- • SQL S06 L03
-- o Multi-line subqueries IN, ANY, ALL
-- Find all reservations for VIP guests
SELECT r.reservation_id, r.check_in_date, r.check_out_date, r.accommodation_price
FROM Reservation r
WHERE r.guest_id IN (
    SELECT guest_id
    FROM Guest
    WHERE guest_type = 'vip'
)
ORDER BY r.check_in_date;

-- Find reservations with accommodation price higher than ANY VIP reservation
SELECT r.reservation_id, g.firstname, g.lastname, r.accommodation_price
FROM Reservation r
JOIN Guest g ON r.guest_id = g.guest_id
WHERE r.accommodation_price > ANY (
    SELECT r2.accommodation_price
    FROM Reservation r2
    JOIN Guest g2 ON r2.guest_id = g2.guest_id
    WHERE g2.guest_type = 'vip'
)
ORDER BY r.accommodation_price DESC;

-- Find reservations with accommodation price higher than ALL standard guest reservations
SELECT r.reservation_id, g.firstname, g.lastname, g.guest_type, r.accommodation_price
FROM Reservation r
JOIN Guest g ON r.guest_id = g.guest_id
WHERE r.accommodation_price > ALL (
    SELECT r2.accommodation_price
    FROM Reservation r2
    JOIN Guest g2 ON r2.guest_id = g2.guest_id
    WHERE g2.guest_type = 'vip'
)
ORDER BY r.accommodation_price;

-- o NULL values in subqueries
-- Find guests who haven't made any reservations
-- Note: This is different from "WHERE reservation_id = NULL" which would always be false
SELECT g.guest_id, g.firstname, g.lastname
FROM Guest g
WHERE NOT EXISTS (
    SELECT 1
    FROM Reservation r
    WHERE r.guest_id = g.guest_id
);
/*********************************************************************************************************/
-- • SQL S06 L04
-- o WITH .. AS() subquery construction

-- Find guests who have not made any reservations
WITH reserved_guests AS (
  SELECT DISTINCT guest_id
  FROM Reservation
)

SELECT firstname || ' ' || lastname AS "Guest without reservation"
FROM Guest
WHERE guest_id NOT IN (
  SELECT guest_id
  FROM reserved_guests
);

/*********************************************************************************************************/
-- • SQL S07 L01
-- o INSERT INTO Tab VALUES()
-- Insert a new room type with all values
INSERT INTO RoomType VALUES
('Prezidentské apartmá', 5);
-- o INSERT INTo Tab (atr, atr) VALUES()
-- Insert a new employee with only required fields
INSERT INTO Employee (firstname, lastname, position, street, city, postal_code, country)
VALUES ('Pavel', 'Malý', 'manažer', 'Zelená 8', 'Brno', '60100', 'ČR');
-- o INSERT INTO Tab AS SELECT …
-- Vložení nových záznamů do tabulky RoomType na základě existujících dat
INSERT INTO RoomType (name, bed_count)
SELECT DISTINCT 'Standardní pokoj', 2
FROM Room
WHERE NOT EXISTS (
    SELECT 1
    FROM RoomType
    WHERE name = 'Standardní pokoj' AND bed_count = 2
);

/*********************************************************************************************************/
-- SQL S07 L02
-- o UPDATE Tab SET atr= …. WHERE condition
-- Update the price of a specific room type
UPDATE RoomTypePriceHistory
SET price_per_night = price_per_night * 1.1
WHERE room_type_id = 5
AND valid_to IS NULL;

-- o DELETE FROM Tab WHERE atr=…
DELETE FROM Feedback
WHERE reservation_id IN (
    SELECT reservation_id
    FROM Reservation
    WHERE status = 'Cancelled'
    AND creation_date < ADD_MONTHS(SYSDATE, -6)
);

/*********************************************************************************************************/
-- • SQL S07 L03
-- o DEFAULT, MERGE, Multi-Table Inserts

-- Insert a new reservation with DEFAULT values for creation_date
INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, 
                         payment_id, status, accommodation_price)
VALUES (1, 5, 3, DEFAULT, SYSDATE + 5, SYSDATE + 10, 5, 'Pending', 3500);

-- simle marge
CREATE TABLE guest_updates (
    email VARCHAR2(100),
    new_phone VARCHAR2(30),
    updated_date DATE
);

-- Insert sample update data
INSERT INTO guest_updates VALUES ('jan.novak@email.cz', '777888999', SYSDATE);
INSERT INTO guest_updates VALUES ('petr.svoboda@email.cz', '666777888', SYSDATE);
INSERT INTO guest_updates VALUES ('new.guest@email.cz', '555666777', SYSDATE);

MERGE INTO Guest g
USING guest_updates u
ON (g.email = u.email)
WHEN MATCHED THEN
    UPDATE SET 
        g.phone = u.new_phone,
        g.notes = 'Phone updated on ' || TO_CHAR(u.updated_date, 'YYYY-MM-DD')
WHEN NOT MATCHED THEN
    INSERT (firstname, lastname, email, phone, birth_date, street, city, postal_code, country, guest_type, registration_date, notes)
    VALUES ('New', 'Guest', u.email, u.new_phone, SYSDATE, 'Unknown', 'Unknown', '00000', 'ČR', 'standard', SYSDATE, 'Added via MERGE');

/*********************************************************************************************************/
-- • SQL S08 L01
-- o Objects in databases – Tables, Indexes, Constraint, View, Sequence, Synonym
-- o CREATE, ALTER, DROP, RENAME, TRUNCATE
-- o CREATE TABLE (atr DAT TYPE, DEFAULT NOT NULL )
-- o ORGANIZATION EXTERNAL, TYPE ORACLE_LOADER, DEFAULT DICTIONARY, ACCESS
-- PARAMETERS, RECORDS DELIMITED BY NEWLINE, FIELDS, LOCATION

-- 1. Table creation
CREATE TABLE GuestArchive (
  guest_id NUMBER PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  email VARCHAR2(100) NOT NULL,
  archived_date DATE DEFAULT SYSDATE
);

CREATE TABLE Invoices (
  invoice_id NUMBER PRIMARY KEY,
  amount NUMBER(10,2) NOT NULL,
  description VARCHAR2(255) NOT NULL,
  invoice_date DATE DEFAULT SYSDATE
);

-- 2. Index creation
-- Single-column index
CREATE INDEX idx_guest_email ON Guest(email);

-- Composite index
CREATE INDEX idx_room_type_number ON Room(room_type_id, room_number);

-- 3. Constraints
-- Check constraint for valid ratings
ALTER TABLE Feedback 
ADD CONSTRAINT chk_rating_range CHECK (rating BETWEEN 1 AND 5);

-- Unique constraint for guest email
ALTER TABLE Guest
ADD CONSTRAINT uq_guest_email UNIQUE (email);

-- 4. Views
-- View for VIP guests
CREATE VIEW vip_guests_view AS
SELECT guest_id, firstname, lastname, email, phone, city
FROM Guest
WHERE guest_type = 'vip';

-- 5. Sequences
-- Sequence for invoice IDs
CREATE SEQUENCE seq_invoice_id
START WITH 1000
INCREMENT BY 1
NOCACHE
NOCYCLE;

-- Using the sequence in an INSERT statement
INSERT INTO Invoices (invoice_id, amount, description)
VALUES (seq_invoice_id.NEXTVAL, 1500, 'Room charge');

-- 6. Synonyms
-- Synonym for frequently accessed table
CREATE SYNONYM active_reservations FOR Reservation;


/*********************************************************************************************************/
-- • SQL S08 L02
-- o TIMESTAMP, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH LOCAL TIMEZONE
CREATE TABLE EventLog (
  event_id NUMBER PRIMARY KEY,
  event_time TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
  event_description VARCHAR2(255) NOT NULL
);

-- Insert events with different time zones
INSERT INTO EventLog (event_id, event_time, event_description)
VALUES (1, TIMESTAMP '2024-06-08 14:30:00 +02:00', 'System backup started');

INSERT INTO EventLog (event_id, event_time, event_description)
VALUES (2, TIMESTAMP '2024-06-08 12:30:00 +00:00', 'Database maintenance');

-- Query showing time zone differences
SELECT 
    event_id, 
    event_description,
    event_time,
    event_time AT TIME ZONE 'UTC' AS utc_time,
    event_time AT TIME ZONE 'America/New_York' AS ny_time
FROM EventLog;

-- Filter events by time, accounting for time zones
SELECT event_id, event_description, event_time
FROM EventLog
WHERE event_time < TIMESTAMP '2024-06-08 13:00:00 +00:00';

-- o INTERVAL YEAR TO MONTH, INTERVAL DAY TO SECOND
SELECT  SYSDATE + delka1 AS "120 months from now", 
        SYSDATE + delka2 AS "3 years 6 months from now"
FROM time_ex4;

-- o CHAR, VARCHAR2, CLOB
-- Example of CHAR and VARCHAR2
CREATE TABLE TextExample (
  fixed_text CHAR(10),
  variable_text VARCHAR2(50),
  large_text CLOB
);

INSERT INTO TextExample (fixed_text, variable_text, large_text)
VALUES ('Fixed', 'Variable length text', 'This is a large text stored in CLOB.');

-- o about NUMBER
-- Example of NUMBER precision and scale
CREATE TABLE NumberExample (
  id NUMBER(10),
  price NUMBER(10,2)
);

INSERT INTO NumberExample (id, price)
VALUES (1, 12345.67);

-- o about BLOB
-- Example of BLOB usage
CREATE TABLE BlobExample (
  id NUMBER PRIMARY KEY,
  image_data BLOB
);

-- Insert a sample BLOB (requires tools or programming language support)
-- Example query to retrieve BLOB data
SELECT id, image_data
FROM BlobExample;

/*********************************************************************************************************/
-- • SQL S08 L03
-- o ALTER TABLE (ADD, MODIFY, DROP), DROP, RENAME
-- Add a new column to Guest table
ALTER TABLE Guest ADD (
    loyalty_points NUMBER DEFAULT 0
);

-- Add multiple columns at once
ALTER TABLE Reservation ADD (
    cancelation_fee NUMBER(10,2),
    special_requests VARCHAR2(500),
    booking_source VARCHAR2(50) DEFAULT 'Website'
);

-- Modify existing column - change data type or constraints
ALTER TABLE Guest MODIFY (
    phone VARCHAR2(20),  -- Change length from whatever it was to 20
    email VARCHAR2(150) NOT NULL  -- Make email non-nullable
);

-- Drop a column
ALTER TABLE Guest DROP COLUMN loyalty_points;

-- Drop multiple columns
ALTER TABLE Reservation DROP (
    cancelation_fee,
    special_requests
);
-- o FLASHBACK TABLE Tab TO BEFORE DROP (view USER_RECYCLEBIN)
-- Drop a table (moves to recyclebin by default)
DROP TABLE GuestArchive;

-- Drop table bypassing the recycle bin
DROP TABLE GuestArchive PURGE;

-- Rename a table
RENAME EventLog TO SystemEventLog;

-- Rename a column
ALTER TABLE Guest RENAME COLUMN notes TO guest_notes;

-- Rename a constraint
ALTER TABLE Guest RENAME CONSTRAINT uq_guest_email TO uq_guest_email_address;
-- First, check what's in the recyclebin
SELECT object_name, original_name, type, droptime
FROM USER_RECYCLEBIN;

-- Restore a dropped table
FLASHBACK TABLE GuestArchive TO BEFORE DROP;

-- Restore with a new name
FLASHBACK TABLE GuestArchive TO BEFORE DROP AS GuestHistory;

-- Purge the recyclebin (removes all dropped objects permanently)
PURGE RECYCLEBIN;

-- Purge a specific object
PURGE TABLE "BIN$AbCdEf123GhI=$0";  -- Use actual system-generated name


-- o DELETE, TRUNCATE
-- DELETE - removes specific rows, can be rolled back, fires triggers
DELETE FROM GuestArchive 
WHERE archived_date < ADD_MONTHS(SYSDATE, -24);

-- DELETE all rows
DELETE FROM GuestArchive;

-- TRUNCATE - removes all rows, cannot be rolled back, faster than DELETE
-- Does not fire triggers, resets storage parameters
TRUNCATE TABLE GuestArchive;

-- TRUNCATE with storage options
TRUNCATE TABLE GuestArchive DROP STORAGE;  -- Release storage space


-- o COMMENT ON TABLE
-- Add comments to document tables and columns
COMMENT ON TABLE Guest IS 'Stores information about hotel guests';

COMMENT ON COLUMN Guest.guest_id IS 'Unique identifier for each guest';
COMMENT ON COLUMN Guest.guest_type IS 'Guest type: standard, vip, etc.';
COMMENT ON COLUMN Guest.registration_date IS 'Date when guest was first registered';

-- View comments
SELECT table_name, comments 
FROM USER_TAB_COMMENTS
WHERE table_name = 'GUEST';

SELECT column_name, comments
FROM USER_COL_COMMENTS
WHERE table_name = 'GUEST';

-- o SET UNUSED
-- Mark a single column as unused (faster than dropping)
ALTER TABLE Guest SET UNUSED COLUMN birth_date;

-- Mark multiple columns as unused
ALTER TABLE Guest SET UNUSED (postal_code, country);

-- Check unused columns
SELECT * FROM USER_UNUSED_COL_TABS;

-- Drop all unused columns (reclaims storage space)
ALTER TABLE Guest DROP UNUSED COLUMNS;

-- Mark column as unused and drop in one operation
ALTER TABLE Guest SET UNUSED COLUMN notes DROP UNUSED COLUMNS;

/*********************************************************************************************************/
-- • SQL S10 L01
-- o CREATE TABLE (NOT NULL AND UNIQUE constraint)
-- Create a new table with explicit NOT NULL and UNIQUE constraints
CREATE TABLE GuestProfile (
  profile_id NUMBER PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  preferred_room_type VARCHAR2(50),
  language_preference VARCHAR2(30) NOT NULL,
  dietary_restrictions VARCHAR2(200),
  email_subscription NUMBER(1) DEFAULT 0 NOT NULL,
  loyalty_card_number VARCHAR2(20) UNIQUE,
  special_instructions VARCHAR2(500),
  CONSTRAINT fk_guest_profile FOREIGN KEY (guest_id) REFERENCES Guest(guest_id)
);

-- o CREATE TABLE Tab AS SELECT …
-- Create an empty table with the same structure as another table
CREATE TABLE GuestArchive2 AS
SELECT * FROM Guest
WHERE 1 = 0;

-- o Own vs. system naming CONSTRAINT conditions

-- Creating constraints with user-defined names
CREATE TABLE HotelService (
  service_id NUMBER,
  service_name VARCHAR2(100),
  price NUMBER(10,2),
  duration_minutes NUMBER,
  is_available NUMBER(1),
  
  -- User-defined constraint names (recommended practice)
  CONSTRAINT pk_hotel_service PRIMARY KEY (service_id),
  CONSTRAINT uq_hotel_service_name UNIQUE (service_name),
  CONSTRAINT nn_service_price CHECK (price IS NOT NULL),
  CONSTRAINT chk_service_price CHECK (price > 0),
  CONSTRAINT chk_service_availability CHECK (is_available IN (0, 1))
);

-- Creating constraints without explicit names (system will generate names)
CREATE TABLE ServiceBooking (
  booking_id NUMBER PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  service_id NUMBER,
  booking_date DATE,
  UNIQUE (guest_id, service_id, booking_date),
  FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  FOREIGN KEY (service_id) REFERENCES HotelService(service_id)
);

-- Viewing constraint names (including system-generated ones)
SELECT constraint_name, constraint_type, table_name, search_condition
FROM user_constraints
WHERE table_name = 'SERVICEBOOKING';

/*********************************************************************************************************/
-- • SQL S10 L02
-- o CONSTRAINT – NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY (atr REFERENCES
-- Add NOT NULL constraint to an existing column
ALTER TABLE Guest MODIFY (phone VARCHAR2(30) NOT NULL);

-- See existing NOT NULL constraints in your Guest table
SELECT column_name, nullable
FROM user_tab_columns
WHERE table_name = 'GUEST' AND nullable = 'N';

-- Add UNIQUE constraint to an existing column
ALTER TABLE Guest ADD CONSTRAINT uq_guest_email UNIQUE (email);

-- Add unique constraint on combination of columns
ALTER TABLE Room ADD CONSTRAINT uq_room_floor_number UNIQUE (room_number);


-- Your tables already have primary keys, but here's how to add one
-- to a new table
CREATE TABLE GuestPreferences (
  guest_id NUMBER,
  preference_type VARCHAR2(50),
  preference_value VARCHAR2(200),
  CONSTRAINT pk_guest_pref PRIMARY KEY (guest_id, preference_type)
);

-- Add PRIMARY KEY to an existing table
ALTER TABLE GuestPreferences ADD CONSTRAINT pk_guest_pref 
PRIMARY KEY (guest_id, preference_type);

-- Your database already has several foreign keys
-- Here's an example from your Reservation table:
CONSTRAINT fk_reservation_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id)

-- Add a new foreign key to an existing table
CREATE TABLE GuestLoyalty (
  loyalty_id NUMBER PRIMARY KEY,
  guest_id NUMBER,
  points NUMBER DEFAULT 0,
  level VARCHAR2(20),
  CONSTRAINT fk_loyalty_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id)
);


-- Tab(atr) ), CHECK

-- Add CHECK constraint to Rating column to ensure it's between 1 and 5
ALTER TABLE Feedback ADD CONSTRAINT chk_rating 
CHECK (rating BETWEEN 1 AND 5);

-- Add CHECK constraint to ensure accommodation price is positive
ALTER TABLE Reservation ADD CONSTRAINT chk_positive_price 
CHECK (accommodation_price > 0);

-- Add CHECK constraint for valid status values
ALTER TABLE Reservation ADD CONSTRAINT chk_valid_status 
CHECK (status IN ('Confirmed', 'Pending', 'Cancelled'));

-- Add CHECK constraint for valid dates (checkout after checkin)
ALTER TABLE Reservation ADD CONSTRAINT chk_valid_dates 
CHECK (check_out_date > check_in_date);


-- o Foreign keys, ON DELETE, ON UPDATE, RESTRICT, CASCADE, etc.
-- Create a table with ON DELETE CASCADE
CREATE TABLE GuestDocument (
  document_id NUMBER PRIMARY KEY,
  guest_id NUMBER,
  document_type VARCHAR2(50),
  upload_date DATE,
  CONSTRAINT fk_doc_guest FOREIGN KEY (guest_id) 
  REFERENCES Guest(guest_id) ON DELETE CASCADE
);

-- When a guest is deleted, all their documents will be deleted automatically
-- Create a table with ON DELETE SET NULL
CREATE TABLE GuestNote (
  note_id NUMBER PRIMARY KEY,
  guest_id NUMBER,
  note_text VARCHAR2(1000),
  created_date DATE,
  CONSTRAINT fk_note_guest FOREIGN KEY (guest_id) 
  REFERENCES Guest(guest_id) ON DELETE SET NULL
);

-- When a guest is deleted, their notes remain but guest_id becomes NULL

/*********************************************************************************************************/
-- • SQL S10 L03
-- o about USER_CONSTRAINTS

-- View the structure of USER_CONSTRAINTS
DESC USER_CONSTRAINTS;

-- Find all constraints for the Reservation table
SELECT constraint_name, constraint_type, search_condition, 
       status, validated, deferrable, deferred
FROM USER_CONSTRAINTS
WHERE table_name = 'RESERVATION';

/*********************************************************************************************************/
-- • SQL S11 L01
-- o CREATE VIEW
-- o about FORCE, NOFORCE
-- o WITCH CHECK OPTION
-- o WITH READ ONLY
-- o about Simple vs. Compex VIEW

-- Simple view of available rooms
CREATE VIEW available_rooms AS
SELECT room_id, room_number, room_type_id
FROM Room
WHERE is_occupied = 0;

-- FORCE view (creates even if referenced table doesn't exist)
CREATE FORCE VIEW future_bookings AS
SELECT guest_id, check_in_date, room_number
FROM ReservationCalendar
WHERE check_in_date > SYSDATE;

-- NOFORCE view (default - only creates if table exists)
CREATE NOFORCE VIEW standard_guests AS
SELECT guest_id, firstname, lastname, email
FROM Guest
WHERE guest_type = 'standard';

-- View with CHECK OPTION to prevent adding non-VIP guests
CREATE VIEW vip_guests AS
SELECT guest_id, firstname, lastname, email, phone
FROM Guest
WHERE guest_type = 'vip'
WITH CHECK OPTION;

-- Read-only view for financial reporting
CREATE VIEW payment_summary AS
SELECT payment_id, total_accommodation, total_expenses, is_paid
FROM Payment
WHERE payment_date IS NOT NULL
WITH READ ONLY;

/*********************************************************************************************************/
-- • SQL S11 L03
-- o INLINE VIEW Subquery in the form of a table SELECT atr FROM (SELECT * FROM Tab)
-- alt_tab

SELECT g.firstname, g.lastname, g.email, r.max_price
FROM Guest g,
    (SELECT guest_id, MAX(accommodation_price) AS max_price
     FROM Reservation
     GROUP BY guest_id) r
WHERE g.guest_id = r.guest_id
AND r.max_price > 3000;

/*********************************************************************************************************/
-- • SQL S12 L01
-- o CREATE SEQUENCE name INCREMENT BY n START WITH m, (NO)MAXVALUE,
-- (NO)MINVALUE, (NO)CYCLE, (NO)CACHE
-- o about ALTER SEQUENCE
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

/*********************************************************************************************************/
-- • SQL S12 L02
-- o CREATE INDEX, PRIMARY KEY, UNIQUE KEY, FOREIGN KEY

CREATE INDEX index_name ON table_name (column_name);

/*********************************************************************************************************/
-- SQL S13 L01
-- Granting permissions to PUBLIC
-- Grant SELECT and UPDATE permissions on the Guest table to all users
GRANT SELECT, UPDATE ON Guest TO PUBLIC;

-- Grant EXECUTE permission on a procedure to a specific user
GRANT EXECUTE ON UpdateGuestInfo TO user_name;

-- Grant INSERT permission on the Reservation table to a specific role
GRANT INSERT ON Reservation TO hotel_staff;

-- Revoking permissions
-- Revoke SELECT permission on the Guest table from PUBLIC
REVOKE SELECT ON Guest FROM PUBLIC;

-- Revoke EXECUTE permission on a procedure from a specific user
REVOKE EXECUTE ON UpdateGuestInfo FROM user_name;

-- Revoke INSERT permission on the Reservation table from a specific role
REVOKE INSERT ON Reservation FROM hotel_staff;

-- Viewing granted permissions
-- Check all grants for the current user
SELECT * FROM USER_TAB_PRIVS;

-- Check all grants for a specific table
SELECT * FROM ALL_TAB_PRIVS WHERE TABLE_NAME = 'GUEST';



/*********************************************************************************************************/
-- SQL S13 L03
-- Regular expressions

-- 1. REGEXP_LIKE: Zkontroluje, zda text odpovídá regulárnímu výrazu
-- Najděte hosty, jejichž email obsahuje doménu "email.cz"
SELECT firstname, lastname, email
FROM Guest
WHERE REGEXP_LIKE(email, '@email\.cz$');

-- 2. REGEXP_REPLACE: Nahrazuje část textu odpovídající regulárnímu výrazu
-- Nahraďte doménu "email.cz" za "example.com" v emailových adresách
SELECT firstname, lastname, REGEXP_REPLACE(email, '@email\.cz$', '@example.com') AS updated_email
FROM Guest;

-- 3. REGEXP_INSTR: Vrací pozici, kde regulární výraz odpovídá textu
-- Najděte pozici "@" v emailových adresách
SELECT firstname, lastname, email, REGEXP_INSTR(email, '@') AS at_position
FROM Guest;

-- 4. REGEXP_SUBSTR: Vrací část textu odpovídající regulárnímu výrazu
-- Extrahujte doménu z emailové adresy
SELECT firstname, lastname, email, REGEXP_SUBSTR(email, '@[^\.]+') AS domain
FROM Guest;

-- 5. REGEXP_COUNT: Počítá počet výskytů regulárního výrazu v textu
-- Spočítejte počet teček v emailových adresách
SELECT firstname, lastname, email, REGEXP_COUNT(email, '\.') AS dot_count
FROM Guest;


/*********************************************************************************************************/
-- SQL S14 L01
-- Transactions, COMMIT, ROLLBACK, SAVEPOINT

-- Start a transaction
BEGIN
  -- Insert a new guest
  INSERT INTO Guest (firstname, lastname, email, phone, birth_date, street, city, postal_code, country, guest_type, registration_date, notes)
  VALUES ('John', 'Doe', 'john.doe@email.com', '123456789', TO_DATE('1985-05-15', 'YYYY-MM-DD'), 'Main Street 1', 'Prague', '11000', 'ČR', 'standard', SYSDATE, null);

  -- Create a savepoint
  SAVEPOINT guest_inserted;

  -- Update the guest's phone number
  UPDATE Guest
  SET phone = '987654321'
  WHERE firstname = 'John' AND lastname = 'Doe';

  -- Rollback to the savepoint
  ROLLBACK TO guest_inserted;

  -- Commit the transaction
  COMMIT;
END;

/*********************************************************************************************************/
-- • SQL S15 L01
-- o Alternative join notation without JOIN with join condition in WHERE
-- o Left and right connection using atrA = atrB (+)


SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date
FROM Guest g, Reservation r
WHERE g.guest_id = r.guest_id;

-- Left outer join using (+)
SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date
FROM Guest g, Reservation r
WHERE g.guest_id = r.guest_id(+);

-- Right outer join using (+)
SELECT r.room_id, r.room_number, rt.name AS room_type_name
FROM Room r, RoomType rt
WHERE r.room_type_id(+) = rt.room_type_id;


/*********************************************************************************************************/
-- • SQL S16 L03
-- o Recapitulation of commands and parameters - complete everything that was not
-- mentioned in the previous points here

-- ??