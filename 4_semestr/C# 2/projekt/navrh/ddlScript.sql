-- Vytvoření databáze
CREATE DATABASE IF NOT EXISTS parkinglot;

-- Vytvoření uživatele
CREATE USER IF NOT EXISTS 'webServer'@'localhost' IDENTIFIED BY 'wabapplogin';

-- Přidělení práv k databázi
GRANT ALL PRIVILEGES ON parkinglot.* TO 'webServer'@'localhost';

-- Pro načtení nových práv
FLUSH PRIVILEGES;

CREATE TABLE ParkingLot (
  parking_lot_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255),
  capacity INT NOT NULL DEFAULT 0;
  latitude DECIMAL(10, 6),
  longitude DECIMAL(10, 6)
);

CREATE TABLE ParkingSpace (
  parking_space_id INT AUTO_INCREMENT PRIMARY KEY,
  parking_lot_id INT,
  space_number INT,
  status VARCHAR(20),
  CONSTRAINT chk_parking_space_status CHECK (status IN ('available', 'occupied', 'under_maintenance'))
);

CREATE TABLE Occupancy (
  occupancy_id INT AUTO_INCREMENT PRIMARY KEY,
  parking_space_id INT,
  license_plate VARCHAR(20),
  start_time DATETIME,
  end_time DATETIME,
  duration INT,
  price DECIMAL(10,2)
);

CREATE TABLE StatusHistory (
  history_id INT AUTO_INCREMENT PRIMARY KEY,
  parking_space_id INT,
  status VARCHAR(20),
  change_time DATETIME,
  CONSTRAINT chk_status_history_status CHECK (status IN ('available', 'occupied', 'under_maintenance'))
);

CREATE TABLE Statistics (
  parking_lot_id INT,
  month DATE,
  completed_parkings INT
);

CREATE TABLE User (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(100) NOT NULL,
  role VARCHAR(50) NOT NULL,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE Car (
    car_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    license_plate VARCHAR(20) NOT NULL,
    brand_model VARCHAR(50),
    color VARCHAR(30),
    FOREIGN KEY (user_id) REFERENCES User(id)
);

CREATE TABLE ParkingHistory (
    history_id INT AUTO_INCREMENT PRIMARY KEY,
    car_id INT NOT NULL,
    parking_lot_id INT NOT NULL,
    arrival_time DATETIME NOT NULL,
    departure_time DATETIME NULL,
    FOREIGN KEY (car_id) REFERENCES Car(car_id),
    FOREIGN KEY (parking_lot_id) REFERENCES ParkingLot(parking_lot_id)
);

ALTER TABLE ParkingSpace ADD FOREIGN KEY (parking_lot_id) REFERENCES ParkingLot (parking_lot_id);

ALTER TABLE Occupancy ADD FOREIGN KEY (parking_space_id) REFERENCES ParkingSpace (parking_space_id);

ALTER TABLE StatusHistory ADD FOREIGN KEY (parking_space_id) REFERENCES ParkingSpace (parking_space_id);

ALTER TABLE Statistics ADD FOREIGN KEY (parking_lot_id) REFERENCES ParkingLot (parking_lot_id);