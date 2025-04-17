-- Vložení adres
INSERT INTO Address (street, city, postal_code, country) VALUES ('Nádražní 123', 'Ostrava', '70200', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Hlavní třída 45', 'Praha', '11000', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Českobratrská 18', 'Brno', '60200', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Zámecká 15', 'Olomouc', '77900', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Stodolní 8', 'Ostrava', '70200', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Sokolská 25', 'Praha', '12000', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Dlouhá 72', 'Plzeň', '30100', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Smetanova 14', 'Brno', '60200', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Háječkova 4', 'Karlovy Vary', '36001', 'Česká republika');
INSERT INTO Address (street, city, postal_code, country) VALUES ('Mariánská 33', 'Liberec', '46001', 'Česká republika');

-- Vložení zaměstnanců
INSERT INTO Employee (firstname, lastname, position, address_id) VALUES ('Jan', 'Novák', 'Recepční', 1);
INSERT INTO Employee (firstname, lastname, position, address_id) VALUES ('Marie', 'Svobodová', 'Manažer', 2);
INSERT INTO Employee (firstname, lastname, position, address_id) VALUES ('Petr', 'Dvořák', 'Pokojská', 3);
INSERT INTO Employee (firstname, lastname, position, address_id) VALUES ('Lucie', 'Černá', 'Recepční', 4);
INSERT INTO Employee (firstname, lastname, position, address_id) VALUES ('Tomáš', 'Procházka', 'Údržbář', 5);

-- Vložení typů pokojů
INSERT INTO RoomType (name, bed_count, price_per_night) VALUES ('Jednolůžkový', 1, 1200.00);
INSERT INTO RoomType (name, bed_count, price_per_night) VALUES ('Dvoulůžkový', 2, 1800.00);
INSERT INTO RoomType (name, bed_count, price_per_night) VALUES ('Apartmá', 2, 3000.00);
INSERT INTO RoomType (name, bed_count, price_per_night) VALUES ('Rodinný', 4, 3500.00);
INSERT INTO RoomType (name, bed_count, price_per_night) VALUES ('Luxusní apartmá', 2, 5000.00);

-- Vložení hostů
INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Karel', 'Malý', 'karel.maly@email.cz', '723456789', TO_DATE('1985-05-15', 'YYYY-MM-DD'), 6, 'Regular', SYSDATE, 'Časté pobyty');

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Anna', 'Veselá', 'anna.vesela@gmail.com', '602123456', TO_DATE('1990-08-21', 'YYYY-MM-DD'), 7, 'VIP', SYSDATE, 'Preferuje klidné pokoje');

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Martin', 'Horák', 'martin.horak@seznam.cz', '777888999', TO_DATE('1978-11-03', 'YYYY-MM-DD'), 8, 'Loyalty', SYSDATE, 'Alergický na ořechy');

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Eva', 'Nováková', 'eva.novakova@email.cz', '608987654', TO_DATE('1995-02-28', 'YYYY-MM-DD'), 9, 'Regular', SYSDATE, NULL);

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Jiří', 'Kovář', 'jiri.kovar@firma.cz', '775321654', TO_DATE('1982-07-17', 'YYYY-MM-DD'), 10, 'VIP', SYSDATE, 'Vždy požaduje extra polštáře');

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Petra', 'Králová', 'petra.kralova@gmail.com', '604111222', TO_DATE('1988-09-12', 'YYYY-MM-DD'), 1, 'Regular', SYSDATE, NULL);

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes) 
VALUES ('Michal', 'Němec', 'michal.nemec@seznam.cz', '731456789', TO_DATE('1975-04-05', 'YYYY-MM-DD'), 2, 'Loyalty', SYSDATE, 'Pravidelné obchodní cesty');

INSERT intO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes)
VALUES ('Jana', 'Benešová', 'janabesova@seznam.cz', '602987654', TO_DATE('1992-12-20', 'YYYY-MM-DD'), 3, 'Regular', SYSDATE, NULL);

INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, registration_date, notes)
VALUES ('David', 'Fiala', 'fiala.david@gmail.com', '604123456', TO_DATE('1980-03-30', 'YYYY-MM-DD'), 4, 'VIP', SYSDATE, 'Preferuje nekuřácké pokoje');

-- Vložení pokojů
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (1, '101', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (1, '102', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (2, '201', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (2, '202', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (3, '301', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (3, '302', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (4, '401', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (4, '402', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (5, '501', 0);
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES (5, '502', 0);

-- Vložení plateb
INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) 
VALUES (3600.00, 500.00, TO_DATE('2023-05-15', 'YYYY-MM-DD'), 1);

INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) 
VALUES (5400.00, 1200.00, TO_DATE('2023-05-18', 'YYYY-MM-DD'), 1);

INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) 
VALUES (9000.00, 2000.00, TO_DATE('2023-06-01', 'YYYY-MM-DD'), 1);

INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) 
VALUES (7000.00, 800.00, TO_DATE('2023-06-05', 'YYYY-MM-DD'), 0);

INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) 
VALUES (10000.00, 3000.00, TO_DATE('2023-06-10', 'YYYY-MM-DD'), 0);

-- Vložení rezervací
INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (1, 1, 1, TO_DATE('2023-05-01', 'YYYY-MM-DD'), TO_DATE('2023-05-15', 'YYYY-MM-DD'), TO_DATE('2023-05-17', 'YYYY-MM-DD'), 1, 'Checked Out');

INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (2, 3, 2, TO_DATE('2023-05-05', 'YYYY-MM-DD'), TO_DATE('2023-05-18', 'YYYY-MM-DD'), TO_DATE('2023-05-21', 'YYYY-MM-DD'), 2, 'Checked Out');

INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (3, 5, 1, TO_DATE('2023-05-10', 'YYYY-MM-DD'), TO_DATE('2023-06-01', 'YYYY-MM-DD'), TO_DATE('2023-06-04', 'YYYY-MM-DD'), 3, 'Checked Out');

INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (4, 7, 4, TO_DATE('2023-05-15', 'YYYY-MM-DD'), TO_DATE('2023-06-15', 'YYYY-MM-DD'), TO_DATE('2023-06-17', 'YYYY-MM-DD'), 4, 'Confirmed');

INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (5, 9, 2, TO_DATE('2023-05-20', 'YYYY-MM-DD'), TO_DATE('2023-06-20', 'YYYY-MM-DD'), TO_DATE('2023-06-22', 'YYYY-MM-DD'), 5, 'Confirmed');

INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status) 
VALUES (6, 2, 3, TO_DATE('2023-05-22', 'YYYY-MM-DD'), TO_DATE('2023-05-25', 'YYYY-MM-DD'), TO_DATE('2023-05-30', 'YYYY-MM-DD'), 1, 'Cancelled');

-- Aktualizace stavu pokojů podle aktivních rezervací
UPDATE Room r
SET r.is_occupied = 1
WHERE r.room_id IN (
    SELECT res.room_id 
    FROM Reservation res 
    WHERE res.status = 'Checked In'
);

-- Commit transakcí
COMMIT;