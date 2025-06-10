-- FULL OUTER JOIN … ON ()
-- Match guests with their feedback, showing all guests and all feedback entries
SELECT g.guest_id, g.firstname, g.lastname, 
       f.feedback_id, f.rating, f.note, f.feedback_date
FROM Guest g
FULL OUTER JOIN Feedback f ON g.guest_id = f.guest_id
ORDER BY g.lastname, f.feedback_date;

-- Count guests by guest type
SELECT guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY guest_type;

-- o Hierarchical querying – tree structure of START WITH, CONNECT BY PRIOR, LEVEL
SELECT LEVEL, employee_id, 
       LPAD(' ', (LEVEL-1)*2) || firstname || ' ' || lastname AS employee_name, 
       position, manager_id
FROM Employee
START WITH manager_id IS NULL
CONNECT BY PRIOR employee_id = manager_id
ORDER SIBLINGS BY employee_id;


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

-- Multi-dimensional analysis of guest counts by city and guest_type
SELECT city, guest_type, COUNT(*) AS guest_count
FROM Guest
GROUP BY CUBE(city, guest_type)
ORDER BY city, guest_type;