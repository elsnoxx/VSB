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

-- natural join
SELECT g.firstname, g.lastname, r.check_in_date, r.check_out_date
FROM Guest g NATURAL JOIN Reservation r;

-- cross join
SELECT g.firstname, g.lastname, r.room_number
FROM Guest g CROSS JOIN Room r;

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


-- • SQL S03 L04
-- o Joining 2x of the same table with renaming (link between superiors and subordinates
-- in one table)
-- Find pairs of employees working in the same city
SELECT 
    e1.employee_id AS emp1_id,
    e1.firstname || ' ' || e1.lastname AS emp1_name,
    e2.employee_id AS emp2_id,
    e2.firstname || ' ' || e2.lastname AS emp2_name,
    e1.city AS shared_city
FROM 
    Employee e1 
JOIN 
    Employee e2 ON e1.city = e2.city
WHERE 
    e1.employee_id < e2.employee_id
ORDER BY 
    e1.city, e1.lastname, e2.lastname;

-- o Hierarchical querying – tree structure of START WITH, CONNECT BY PRIOR, LEVEL
-- dive

-- todo

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
ORDER BY guest_count DESC

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
    WHERE guest_type = 'vip' AND city = 'Praha'
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

-- Find rooms that have never been booked
SELECT r.room_id, r.room_number, rt.name AS room_type
FROM Room r
JOIN RoomType rt ON r.room_type_id = rt.room_type_id
WHERE NOT EXISTS (
    SELECT 1
    FROM Reservation res
    WHERE res.room_id = r.room_id
)
ORDER BY r.room_number;


-- • SQL S06 L02
-- o One-line subqueries
-- Calculate how much each reservation differs from the average accommodation price
SELECT r.reservation_id, r.guest_id, r.accommodation_price, 
       r.accommodation_price - (SELECT AVG(accommodation_price) FROM Reservation) AS price_difference
FROM Reservation r
ORDER BY price_difference DESC;


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
    WHERE g2.guest_type = 'standard'
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

-- • SQL S06 L04
-- o WITH .. AS() subquery construction


-- • SQL S07 L01
-- o INSERT INTO Tab VALUES()
-- Insert a new room type with all values
INSERT INTO RoomType VALUES
(21, 'Prezidentské apartmá', 5);
-- o INSERT INTo Tab (atr, atr) VALUES()
-- Insert a new employee with only required fields
INSERT INTO Employee (firstname, lastname, position, street, city, postal_code, country)
VALUES ('Pavel', 'Malý', 'manažer', 'Zelená 8', 'Brno', '60100', 'ČR');
-- o INSERT INTO Tab AS SELECT …
???

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

-- • SQL S08 L01
-- o Objects in databases – Tables, Indexes, Constraint, View, Sequence, Synonym
-- o CREATE, ALTER, DROP, RENAME, TRUNCATE
-- o CREATE TABLE (atr DAT TYPE, DEFAULT NOT NULL )
-- o ORGANIZATION EXTERNAL, TYPE ORACLE_LOADER, DEFAULT DICTIONARY, ACCESS
-- PARAMETERS, RECORDS DELIMITED BY NEWLINE, FIELDS, LOCATION

-- Basic table creation
CREATE TABLE GuestArchive (
  guest_id NUMBER PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  email VARCHAR2(100) NOT NULL,
  archived_date DATE DEFAULT SYSDATE
);

-- Create an index on the guest email column
CREATE INDEX idx_guest_email ON Guest(email);

-- Create a composite index on multiple columns
CREATE INDEX idx_room_type_number ON Room(room_type_id, room_number);

-- Add a check constraint to ensure valid ratings
ALTER TABLE Feedback 
ADD CONSTRAINT chk_rating_range CHECK (rating BETWEEN 1 AND 5);

-- Add a unique constraint
ALTER TABLE Guest
ADD CONSTRAINT uq_guest_email UNIQUE (email);

-- Create a view of VIP guests
CREATE VIEW vip_guests_view AS
SELECT guest_id, firstname, lastname, email, phone, city
FROM Guest
WHERE guest_type = 'vip'

-- Create a sequence for a new table
CREATE SEQUENCE seq_invoice_id
START WITH 1000
INCREMENT BY 1
NOCACHE
NOCYCLE;

-- Using the sequence in an INSERT statement
INSERT INTO Invoices (invoice_id, amount, description)
VALUES (seq_invoice_id.NEXTVAL, 1500, 'Room charge');

-- Create a synonym for a frequently accessed table
CREATE SYNONYM active_reservations FOR 
  (SELECT * FROM Reservation WHERE status = 'Confirmed');



-- • SQL S08 L02
CREATE TABLE EventLog (
  event_id NUMBER PRIMARY KEY,
  event_time TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
  event_description VARCHAR2(255) NOT NULL
);
-- o TIMESTAMP, TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH LOCAL TIMEZONE
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
WHERE event_time > TIMESTAMP '2024-06-08 13:00:00 +00:00';
-- o INTERVAL YEAT TO MONTH, INTERVAL DAY TO SECOND
-- o CHAR, VARCHAR2, CLOB
-- o about NUMBER
-- o about BLOB

-- todo

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
CREATE TABLE GuestArchive AS
SELECT * FROM Guest
WHERE 1 = 0;  -- This condition is always false, so no rows are selected

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
  booking_id NUMBER PRIMARY KEY,  -- System will name this constraint
  guest_id NUMBER NOT NULL,       -- NOT NULL constraints don't get names
  service_id NUMBER,
  booking_date DATE,
  UNIQUE (guest_id, service_id, booking_date), -- System will name this constraint
  FOREIGN KEY (guest_id) REFERENCES Guest(guest_id), -- System will name this constraint
  FOREIGN KEY (service_id) REFERENCES HotelService(service_id) -- System will name this
);

-- Viewing constraint names (including system-generated ones)
SELECT constraint_name, constraint_type, table_name, search_condition
FROM user_constraints
WHERE table_name = 'SERVICEBOOKING';


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


-- • SQL S10 L03
-- o about USER_CONSTRAINTS

-- View the structure of USER_CONSTRAINTS
DESC USER_CONSTRAINTS;

-- Find all constraints for the Reservation table
SELECT constraint_name, constraint_type, search_condition, 
       status, validated, deferrable, deferred
FROM USER_CONSTRAINTS
WHERE table_name = 'RESERVATION';


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
FROM ReservationCalendar  -- This table doesn't exist yet
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


-- • SQL S11 L03
-- o INLINE VIEW Subquery in the form of a table SELECT atr FROM (SELECT * FROM Tab)
-- alt_tab


-- • SQL S12 L01
-- o CREATE SEQUENCE name INCREMENT BY n START WITH m, (NO)MAXVALUE,
-- (NO)MINVALUE, (NO)CYCLE, (NO)CACHE
-- o about ALTER SEQUENCE


-- • SQL S12 L02
-- o CREATE INDEX, PRIMARY KEY, UNIQUE KEY, FOREIGN KEY

CREATE INDEX index_name ON table_name (column_name);

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