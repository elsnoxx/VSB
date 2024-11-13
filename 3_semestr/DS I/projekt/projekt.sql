-- Drop tables if they exist
DROP TABLE IF EXISTS Reservation;
DROP TABLE IF EXISTS Payment;
DROP TABLE IF EXISTS Room;
DROP TABLE IF EXISTS RoomType;
DROP TABLE IF EXISTS Guest;
DROP TABLE IF EXISTS Employee;
DROP TABLE IF EXISTS Address;
GO

-- Create Address table
CREATE TABLE Address (
  address_id int PRIMARY KEY IDENTITY(1, 1),
  street varchar(255) NOT NULL,
  city varchar(100) NOT NULL,
  postal_code char(10) NOT NULL,
  country varchar(100) NOT NULL
);
GO

-- Create Guest table
CREATE TABLE Guest (
  guest_id int PRIMARY KEY IDENTITY(1, 1),
  firstname varchar(100) NOT NULL,
  lastname varchar(100) NOT NULL,
  email varchar(100) NOT NULL,
  phone varchar(15) NOT NULL,
  birth_date date,
  address_id int, -- Foreign key to Address table
  guest_type varchar(50),
  registration_date date DEFAULT GETDATE(),
  notes text,
  FOREIGN KEY (address_id) REFERENCES Address(address_id)
);
GO

-- Create Employee table
CREATE TABLE Employee (
  employee_id int PRIMARY KEY IDENTITY(1, 1),
  firstname varchar(100) NOT NULL,
  lastname varchar(100) NOT NULL,
  position varchar(50) NOT NULL,
  address_id int, -- Foreign key to Address table
  FOREIGN KEY (address_id) REFERENCES Address(address_id)
);
GO

-- Create RoomType table
CREATE TABLE RoomType (
  room_type_id int PRIMARY KEY IDENTITY(1, 1),
  name varchar(50) NOT NULL,
  bed_count int NOT NULL,
  price_per_night decimal(10, 2) NOT NULL
);
GO

-- Create Room table
CREATE TABLE Room (
  room_id int PRIMARY KEY IDENTITY(1, 1),
  room_type_id int NOT NULL,
  room_number varchar(10) NOT NULL,
  is_occupied bit NOT NULL,
  FOREIGN KEY (room_type_id) REFERENCES RoomType (room_type_id)
);
GO

-- Create Payment table (do not add foreign key yet)
CREATE TABLE Payment (
  payment_id int PRIMARY KEY IDENTITY(1, 1),
  total_accommodation decimal(10, 2) NOT NULL,
  total_expenses decimal(10, 2) NOT NULL,
  payment_date date NOT NULL,
  is_paid bit NOT NULL
);
GO

-- Create Reservation table with foreign key to Payment
CREATE TABLE Reservation (
  reservation_id int PRIMARY KEY IDENTITY(1, 1),
  guest_id int NOT NULL,
  room_id int NOT NULL,
  employee_id int NOT NULL,
  creation_date date NOT NULL,
  check_in_date date NOT NULL,
  check_out_date date,
  payment_id int, -- Foreign key to Payment table
  status varchar(20) NOT NULL,
  FOREIGN KEY (guest_id) REFERENCES Guest (guest_id),
  FOREIGN KEY (room_id) REFERENCES Room (room_id),
  FOREIGN KEY (employee_id) REFERENCES Employee (employee_id),
  FOREIGN KEY (payment_id) REFERENCES Payment (payment_id) -- This should come after Payment is created
);
GO



-- Insert into Address table
INSERT INTO Address (street, city, postal_code, country)
VALUES
('Hlavní 123', 'Praha', '11000', 'Česká republika'),
('Masarykova 45', 'Brno', '60200', 'Česká republika'),
('Jiráskova 7', 'Ostrava', '70030', 'Česká republika'),
('Křižíkova 32', 'Plzeň', '30100', 'Česká republika'),
('Bělohorská 98', 'Liberec', '46001', 'Česká republika'),
('Vinohradská 214', 'Praha', '13000', 'Česká republika'),
('Kapitána Nálepky 740/6', 'Havířov', '73601', 'Česká republika'),
('U Hlavního nádraží 45', 'Brno', '60200', 'Česká republika'),
('Horní 7', 'Ostrava', '70030', 'Česká republika'),
('Na Příkopě 32', 'Plzeň', '30100', 'Česká republika'),
('Stodolní 15', 'Ostrava', '70200', 'Česká republika'),
('Sokolská 122', 'Ostrava', '70300', 'Česká republika');
GO

-- Insert into Guest table
INSERT INTO Guest (firstname, lastname, email, phone, birth_date, address_id, guest_type, notes)
VALUES 
('Anna', 'Nováková', 'anna.novakova@email.com', '603123456', '1985-06-15', 1, 'Regular', 'Častý host, upřednostňuje pokoj s výhledem na moře'),
('Jan', 'Kovář', 'jan.kovar@email.com', '777654321', '1990-02-23', 2, 'VIP', 'Pravidelný VIP host, preferuje apartmá se stravováním v pokoji'),
('Eva', 'Svobodová', 'eva.svobodova@email.com', '608987654', '1978-11-03', 3, 'Regular', 'Host s preferencí wellness služeb, dlouhodobý pobyt'),
('Petr', 'Dvořák', 'petr.dvorak@email.com', '702123789', '1982-09-20', 4, 'Business', 'Častý host na služebních cestách, vyžaduje rychlý internet'),
('Lucie', 'Králová', 'lucie.kralova@email.com', '605456123', '1995-04-12', 5, 'Regular', 'Nový host, zájem o wellness balíčky'),
('Martin', 'Horák', 'martin.horak@email.com', '732567890', '1988-07-19', 6, 'VIP', 'VIP host s individuálními požadavky na stravování'),
('Richard', 'Ficek', 'richard.ficek.st@vsb.cz', '730585034', '2001-01-15', 7 ,'VIP', '');
GO

-- Insert into Employee table
INSERT INTO Employee (firstname, lastname, position, address_id)
VALUES 
('Jan', 'Novotný', 'Recepční', 8),
('Petra', 'Krejčí', 'Recepční', 9),
('Marek', 'Kovář', 'Recepční', 10),
('Alena', 'Havlíková', 'Manažer', 11),
('Tomáš', 'Procházka', 'Majitel', 12);
GO

-- Insert into RoomType table
INSERT INTO RoomType (name, bed_count, price_per_night)
VALUES
('Dvoulůžkový', 2, 1200.00),
('Třílůžkový', 3, 1800.00),
('Čtyřlůžkový', 4, 2400.00),
('Šestilůžkový', 6, 3600.00);
GO

-- Insert into Room table
-- Patro 1
INSERT INTO Room (room_type_id, room_number, is_occupied)
VALUES
(1, '101', 0),
(1, '102', 0),
(1, '103', 1),
(2, '104', 0),
(2, '105', 0),
(3, '106', 0),
(4, '107', 0);

-- Patro 2
INSERT INTO Room (room_type_id, room_number, is_occupied)
VALUES
(1, '201', 1),
(1, '202', 0),
(1, '203', 0),
(2, '204', 1),
(2, '205', 0),
(3, '206', 0),
(3, '207', 0);

-- Patro 3
INSERT INTO Room (room_type_id, room_number, is_occupied)
VALUES
(1, '301', 0),
(1, '302', 0),
(1, '303', 0),
(2, '304', 0),
(2, '305', 0),
(3, '306', 1),
(4, '307', 0); 
GO

-- Insert into Payment table
INSERT INTO Payment (total_accommodation, total_expenses, payment_date, is_paid)
VALUES
(6000.00, 0.00, GETDATE(), 1),
(4800.00, 200.00, GETDATE(), 1),
(0.00, 0.00, GETDATE(), 0),
(0.00, 0.00, GETDATE(), 0),
(0.00, 0.00, GETDATE(), 0),
(0.00, 0.00, GETDATE(), 0),
(0.00, 0.00, GETDATE(), 1);
GO

-- Insert into Reservation table
INSERT INTO Reservation (guest_id, room_id, employee_id, creation_date, check_in_date, check_out_date, status, payment_id)
VALUES 
(1, 1, 1, GETDATE(), '2024-11-01', '2024-11-05', 'Confirmed', 1),
(2, 11, 2, GETDATE(), '2024-11-02', NULL, 'Checked In', 3),
(3, 6, 3, GETDATE(), '2024-11-03', '2024-11-06', 'Checked Out', 2),
(4, 3, 1, GETDATE(), '2024-10-30', NULL, 'Checked In', 4),
(5, 8, 2, GETDATE(), '2024-11-07', NULL, 'Checked In', 5),
(6, 1, 1, GETDATE(), '2024-11-07', NULL, 'Checked In', 6),
(1, 2, 1, GETDATE(), '2024-10-28', NULL, 'Cancelled', NULL),
(2, 8, 2, GETDATE(), '2024-10-30', NULL, 'Cancelled', NULL);
GO



