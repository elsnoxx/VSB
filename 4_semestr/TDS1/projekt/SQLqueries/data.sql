-- Guest
INSERT INTO Guest (firstname, lastname, email, phone, birth_date, street, city, postal_code, country, guest_type, registration_date, notes) VALUES
('Jan', 'Novák', 'jan.novak@email.cz', '123456789', '1990-01-01', 'Hlavní 1', 'Praha', '11000', 'ČR', 'standard', GETDATE(), NULL),
('Petr', 'Svoboda', 'petr.svoboda@email.cz', '234567890', '1985-02-02', 'Vedlejší 2', 'Brno', '60200', 'ČR', 'vip', GETDATE(), NULL),
('Eva', 'Dvořáková', 'eva.dvorakova@email.cz', '345678901', '1992-03-03', 'Třetí 3', 'Ostrava', '70030', 'ČR', 'standard', GETDATE(), NULL),
('Lucie', 'Králová', 'lucie.kralova@email.cz', '456789012', '1992-03-03', 'Čtvrtá 4', 'Plzeň', '30100', 'ČR', 'standard', GETDATE(), NULL),
('Martin', 'Procházka', 'martin.prochazka@email.cz', '567890123', '1992-03-03', 'Pátá 5', 'Liberec', '46001', 'ČR', 'vip', GETDATE(), NULL),
('Jana', 'Kučerová', 'jana.kucerova@email.cz', '678901234', '1992-03-03', 'Šestá 6', 'Olomouc', '77900', 'ČR', 'standard', GETDATE(), NULL),
('Tomáš', 'Urban', 'tomas.urban@email.cz', '789012345', '1992-03-03', 'Sedmá 7', 'Zlín', '76001', 'ČR', 'standard', GETDATE(), NULL),
('Barbora', 'Benešová', 'barbora.benesova@email.cz', '890123456', '1992-03-03', 'Osmá 8', 'Pardubice', '53002', 'ČR', 'vip', GETDATE(), NULL),
('David', 'Veselý', 'david.vesely@email.cz', '901234567', '1992-03-03', 'Devátá 9', 'Hradec Králové', '50003', 'ČR', 'standard', GETDATE(), NULL),
('Veronika', 'Horáková', 'veronika.horakova@email.cz', '123450987', '1992-03-03', 'Desátá 10', 'Ústí nad Labem', '40001', 'ČR', 'standard', GETDATE(), NULL),
('Filip', 'Němec', 'filip.nemec@email.cz', '234561098', '1992-03-03', 'Jedenáctá 11', 'České Budějovice', '37001', 'ČR', 'vip', GETDATE(), NULL),
('Tereza', 'Marek', 'tereza.marek@email.cz', '345672109', '1992-03-03', 'Dvanáctá 12', 'Jihlava', '58601', 'ČR', 'standard', GETDATE(), NULL),
('Ondřej', 'Pokorný', 'ondrej.pokorny@email.cz', '456783210', '1992-03-03', 'Třináctá 13', 'Karlovy Vary', '36001', 'ČR', 'standard', GETDATE(), NULL),
('Kateřina', 'Pospíšilová', 'katerina.pospisilova@email.cz', '567894321', '1992-03-03', 'Čtrnáctá 14', 'Teplice', '41501', 'ČR', 'vip', GETDATE(), NULL),
('Jakub', 'Hájek', 'jakub.hajek@email.cz', '678905432', '1992-03-03', 'Patnáctá 15', 'Opava', '74601', 'ČR', 'standard', GETDATE(), NULL),
('Michaela', 'Sedláčková', 'michaela.sedlackova@email.cz', '789016543', '1992-03-03', 'Šestnáctá 16', 'Trutnov', '54101', 'ČR', 'standard', GETDATE(), NULL),
('Roman', 'Doležal', 'roman.dolezal@email.cz', '890127654', '1992-03-03', 'Sedmnáctá 17', 'Příbram', '26101', 'ČR', 'vip', GETDATE(), NULL),
('Simona', 'Zemanová', 'simona.zemanova@email.cz', '901238765', '1992-03-03', 'Osmnáctá 18', 'Mladá Boleslav', '29301', 'ČR', 'standard', GETDATE(), NULL),
('Radek', 'Kolář', 'radek.kolar@email.cz', '123459876', '1992-03-03', 'Devatenáctá 19', 'Kladno', '27201', 'ČR', 'standard', GETDATE(), NULL),
('Alena', 'Navrátilová', 'alena.navratilova@email.cz', '234560987', '1992-03-03', 'Dvacátá 20', 'Znojmo', '66902', 'ČR', 'vip', GETDATE(), NULL);
-- Employee
INSERT INTO Employee (firstname, lastname, position, street, city, postal_code, country) VALUES
('Petr', 'Svoboda', 'recepční', 'Vedlejší 2', 'Praha', '11000', 'ČR'),
('Jana', 'Novotná', 'recepční', 'Hlavní 3', 'Brno', '60200', 'ČR'),
('Martin', 'Král', 'údržbář', 'Třetí 4', 'Ostrava', '70030', 'ČR'),
('Eva', 'Bartošová', 'pokojská', 'Čtvrtá 5', 'Plzeň', '30100', 'ČR'),
('Tomáš', 'Dvořák', 'recepční', 'Pátá 6', 'Liberec', '46001', 'ČR'),
('Lucie', 'Kovářová', 'pokojská', 'Šestá 7', 'Olomouc', '77900', 'ČR'),
('David', 'Procházka', 'údržbář', 'Sedmá 8', 'Zlín', '76001', 'ČR'),
('Barbora', 'Urbanová', 'recepční', 'Osmá 9', 'Pardubice', '53002', 'ČR'),
('Filip', 'Horák', 'recepční', 'Devátá 10', 'Hradec Králové', '50003', 'ČR'),
('Veronika', 'Němcová', 'pokojská', 'Desátá 11', 'Ústí nad Labem', '40001', 'ČR'),
('Ondřej', 'Marek', 'údržbář', 'Jedenáctá 12', 'České Budějovice', '37001', 'ČR'),
('Tereza', 'Pokorná', 'recepční', 'Dvanáctá 13', 'Jihlava', '58601', 'ČR'),
('Roman', 'Pospíšil', 'recepční', 'Třináctá 14', 'Karlovy Vary', '36001', 'ČR'),
('Kateřina', 'Hájek', 'pokojská', 'Čtrnáctá 15', 'Teplice', '41501', 'ČR'),
('Jakub', 'Sedláček', 'údržbář', 'Patnáctá 16', 'Opava', '74601', 'ČR'),
('Michaela', 'Doležalová', 'recepční', 'Šestnáctá 17', 'Trutnov', '54101', 'ČR'),
('Radek', 'Zeman', 'recepční', 'Sedmnáctá 18', 'Příbram', '26101', 'ČR'),
('Simona', 'Kolářová', 'pokojská', 'Osmnáctá 19', 'Mladá Boleslav', '29301', 'ČR'),
('Alena', 'Navrátilová', 'recepční', 'Devatenáctá 20', 'Kladno', '27201', 'ČR'),
('Jan', 'Beneš', 'údržbář', 'Dvacátá 21', 'Znojmo', '66902', 'ČR');

-- RoomType
INSERT INTO RoomType (name, bed_count) VALUES
('Jednolůžkový', 1), ('Dvoulůžkový', 2), ('Třílůžkový', 3), ('Apartmán', 4),
('Jednolůžkový Deluxe', 1), ('Dvoulůžkový Deluxe', 2), ('Třílůžkový Deluxe', 3), ('Apartmán Deluxe', 4),
('Jednolůžkový Economy', 1), ('Dvoulůžkový Economy', 2), ('Třílůžkový Economy', 3), ('Apartmán Economy', 4),
('Jednolůžkový Standard', 1), ('Dvoulůžkový Standard', 2), ('Třílůžkový Standard', 3), ('Apartmán Standard', 4),
('Jednolůžkový Premium', 1), ('Dvoulůžkový Premium', 2), ('Třílůžkový Premium', 3), ('Apartmán Premium', 4);

-- Room
INSERT INTO Room (room_type_id, room_number, is_occupied) VALUES
(1, '101', 0), (2, '102', 0), (3, '103', 0), (4, '104', 0), (5, '105', 0),
(6, '106', 0), (7, '107', 0), (8, '108', 0), (9, '109', 0), (10, '110', 0),
(11, '111', 0), (12, '112', 0), (13, '113', 0), (14, '114', 0), (15, '115', 0),
(16, '116', 0), (17, '117', 0), (18, '118', 0), (19, '119', 0), (20, '120', 0);

-- Payment
INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid) VALUES
(2000, 300, '2024-06-05', 1), (3200, 500, '2024-06-10', 1), (1500, 200, '2024-06-15', 0), (4000, 800, '2024-06-20', 1), (2500, 400, '2024-06-25', 0),
(1800, 250, '2024-07-01', 1), (3500, 600, '2024-07-05', 1), (2200, 350, '2024-07-10', 0), (4100, 900, '2024-07-15', 1), (2600, 450, '2024-07-20', 0),
(1900, 270, '2024-07-25', 1), (3600, 650, '2024-07-30', 1), (2300, 370, '2024-08-04', 0), (4200, 950, '2024-08-09', 1), (2700, 470, '2024-08-14', 0),
(1950, 280, '2024-08-19', 1), (3650, 660, '2024-08-24', 1), (2350, 380, '2024-08-29', 0), (4300, 980, '2024-09-03', 1), (2800, 490, '2024-09-08', 0);


-- Service
INSERT INTO Service (name, description) VALUES
('Snídaně', 'Snídaně formou bufetu'), ('Wellness', 'Vstup do wellness centra'), ('Parkování', 'Parkování u hotelu'), ('Wi-Fi', 'Bezdrátové připojení k internetu'),
('Bazén', 'Vstup do bazénu'), ('Fitness', 'Vstup do fitness centra'), ('Masáže', 'Relaxační masáže'), ('Bar', 'Hotelový bar'),
('Praní prádla', 'Praní a žehlení prádla'), ('Půjčení kola', 'Půjčení jízdního kola'), ('Půjčení auta', 'Půjčení automobilu'), ('Dětský koutek', 'Herní koutek pro děti'),
('Konferenční místnost', 'Pronájem konferenční místnosti'), ('Room service', 'Pokojová služba'), ('Transfer', 'Transfer na letiště'), ('Zvířata', 'Pobyt se zvířetem'),
('Minibar', 'Minibar na pokoji'), ('Sauna', 'Vstup do sauny'), ('Solárium', 'Vstup do solária'), ('Tenis', 'Tenisový kurt');



-- RoomTypePriceHistory
INSERT INTO RoomTypePriceHistory (room_type_id, price_per_night, valid_from, valid_to) VALUES
(1, 500, CAST('2024-01-01' AS DATE), NULL), (2, 800, CAST('2024-01-01' AS DATE), NULL), (3, 1200, CAST('2024-01-01' AS DATE), NULL), (4, 2000, CAST('2024-01-01' AS DATE), NULL),
(5, 600, CAST('2024-01-01' AS DATE), NULL), (6, 900, CAST('2024-01-01' AS DATE), NULL), (7, 1300, CAST('2024-01-01' AS DATE), NULL), (8, 2100, CAST('2024-01-01' AS DATE), NULL),
(9, 550, CAST('2024-01-01' AS DATE), NULL), (10, 850, CAST('2024-01-01' AS DATE), NULL), (11, 1250, CAST('2024-01-01' AS DATE), NULL), (12, 2050, CAST('2024-01-01' AS DATE), NULL),
(13, 520, CAST('2024-01-01' AS DATE), NULL), (14, 820, CAST('2024-01-01' AS DATE), NULL), (15, 1220, CAST('2024-01-01' AS DATE), NULL), (16, 2020, CAST('2024-01-01' AS DATE), NULL),
(17, 650, CAST('2024-01-01' AS DATE), NULL), (18, 950, CAST('2024-01-01' AS DATE), NULL), (19, 1350, CAST('2024-01-01' AS DATE), NULL), (20, 2150, CAST('2024-01-01' AS DATE), NULL);

-- ServicePriceHistory
INSERT INTO ServicePriceHistory (service_id, price, valid_from, valid_to) VALUES
(1, 100, CAST('2024-01-01' AS DATE), NULL), (2, 200, CAST('2024-01-01' AS DATE), NULL), (3, 150, CAST('2024-01-01' AS DATE), NULL), (4, 50, CAST('2024-01-01' AS DATE), NULL),
(5, 120, CAST('2024-01-01' AS DATE), NULL), (6, 180, CAST('2024-01-01' AS DATE), NULL), (7, 250, CAST('2024-01-01' AS DATE), NULL), (8, 90, CAST('2024-01-01' AS DATE), NULL),
(9, 110, CAST('2024-01-01' AS DATE), NULL), (10, 130, CAST('2024-01-01' AS DATE), NULL), (11, 170, CAST('2024-01-01' AS DATE), NULL), (12, 140, CAST('2024-01-01' AS DATE), NULL),
(13, 160, CAST('2024-01-01' AS DATE), NULL), (14, 210, CAST('2024-01-01' AS DATE), NULL), (15, 80, CAST('2024-01-01' AS DATE), NULL), (16, 60, CAST('2024-01-01' AS DATE), NULL),
(17, 190, CAST('2024-01-01' AS DATE), NULL), (18, 220, CAST('2024-01-01' AS DATE), NULL), (19, 230, CAST('2024-01-01' AS DATE), NULL), (20, 240, CAST('2024-01-01' AS DATE), NULL);

-- Feedback
INSERT INTO Feedback (guest_id, reservation_id, rating, comment, feedback_date) VALUES
(1, 1, 5, 'Vše v pořádku', GETDATE()), (2, 2, 4, 'Pěkný pobyt', GETDATE()), (3, 3, 3, 'Průměrné služby', GETDATE()), (4, 4, 5, 'Výborné jídlo', GETDATE()), (5, 5, 2, 'Pokoj nebyl čistý', GETDATE()),
(6, 6, 4, 'Příjemný personál', GETDATE()), (7, 7, 5, 'Skvělá lokalita', GETDATE()), (8, 8, 3, 'Moc hlučno', GETDATE()), (9, 9, 4, 'Dobré snídaně', GETDATE()), (10, 10, 5, 'Doporučuji', GETDATE()),
(11, 11, 4, 'Hezký pokoj', GETDATE()), (12, 12, 5, 'Super wellness', GETDATE()), (13, 13, 3, 'Malý pokoj', GETDATE()), (14, 14, 4, 'Dobrá cena', GETDATE()), (15, 15, 5, 'Vše v pořádku', GETDATE()),
(16, 16, 4, 'Pěkný výhled', GETDATE()), (17, 17, 5, 'Výborné služby', GETDATE()), (18, 18, 3, 'Průměrné jídlo', GETDATE()), (19, 19, 4, 'Příjemné prostředí', GETDATE()), (20, 20, 5, 'Doporučuji', GETDATE());

-- ServiceUsage
INSERT INTO ServiceUsage (reservation_id, service_id, quantity, total_price) VALUES
(1, 1, 2, 300), (2, 2, 1, 500), (3, 3, 1, 200), (4, 4, 1, 100), (5, 5, 2, 400),
(6, 6, 1, 250), (7, 7, 1, 350), (8, 8, 2, 300), (9, 9, 1, 150), (10, 10, 1, 200),
(11, 11, 1, 500), (12, 12, 2, 400), (13, 13, 1, 600), (14, 14, 1, 250), (15, 15, 1, 300),
(16, 16, 2, 400), (17, 17, 1, 350), (18, 18, 1, 200), (19, 19, 1, 150), (20, 20, 1, 200);

-- Reservation
INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, payment_id, status, accommodation_price) VALUES
(1, 1, 1, GETDATE(), CAST('2024-06-01' AS DATE), CAST('2024-06-05' AS DATE), 1, 'Confirmed', 2000),
(2, 2, 2, GETDATE(), CAST('2024-06-02' AS DATE), CAST('2024-06-06' AS DATE), 2, 'Confirmed', 3200),
(3, 3, 3, GETDATE(), CAST('2024-06-03' AS DATE), CAST('2024-06-07' AS DATE), 3, 'Pending', 1500),
(4, 4, 4, GETDATE(), CAST('2024-06-04' AS DATE), CAST('2024-06-08' AS DATE), 4, 'Confirmed', 4000),
(5, 5, 5, GETDATE(), CAST('2024-06-05' AS DATE), CAST('2024-06-09' AS DATE), 5, 'Cancelled', 2500),
(6, 6, 6, GETDATE(), CAST('2024-06-06' AS DATE), CAST('2024-06-10' AS DATE), 6, 'Confirmed', 1800),
(7, 7, 7, GETDATE(), CAST('2024-06-07' AS DATE), CAST('2024-06-11' AS DATE), 7, 'Confirmed', 3500),
(8, 8, 8, GETDATE(), CAST('2024-06-08' AS DATE), CAST('2024-06-12' AS DATE), 8, 'Pending', 2200),
(9, 9, 9, GETDATE(), CAST('2024-06-09' AS DATE), CAST('2024-06-13' AS DATE), 9, 'Confirmed', 4100),
(10, 10, 10, GETDATE(), CAST('2024-06-10' AS DATE), CAST('2024-06-14' AS DATE), 10, 'Confirmed', 2600),
(11, 11, 11, GETDATE(), CAST('2024-06-11' AS DATE), CAST('2024-06-15' AS DATE), 11, 'Pending', 1900),
(12, 12, 12, GETDATE(), CAST('2024-06-12' AS DATE), CAST('2024-06-16' AS DATE), 12, 'Confirmed', 3600),
(13, 13, 13, GETDATE(), CAST('2024-06-13' AS DATE), CAST('2024-06-17' AS DATE), 13, 'Confirmed', 2300),
(14, 14, 14, GETDATE(), CAST('2024-06-14' AS DATE), CAST('2024-06-18' AS DATE), 14, 'Pending', 4200),
(15, 15, 15, GETDATE(), CAST('2024-06-15' AS DATE), CAST('2024-06-19' AS DATE), 15, 'Confirmed', 2700),
(16, 16, 16, GETDATE(), CAST('2024-06-16' AS DATE), CAST('2024-06-20' AS DATE), 16, 'Confirmed', 1950),
(17, 17, 17, GETDATE(), CAST('2024-06-17' AS DATE), CAST('2024-06-21' AS DATE), 17, 'Pending', 3650),
(18, 18, 18, GETDATE(), CAST('2024-06-18' AS DATE), CAST('2024-06-22' AS DATE), 18, 'Confirmed', 2350),
(19, 19, 19, GETDATE(), CAST('2024-06-19' AS DATE), CAST('2024-06-23' AS DATE), 19, 'Confirmed', 4300),
(20, 20, 20, GETDATE(), CAST('2024-06-20' AS DATE), CAST('2024-06-24' AS DATE), 20, 'Cancelled', 2800);