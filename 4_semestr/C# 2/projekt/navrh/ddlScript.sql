CREATE TABLE ParkingLot (
  parking_lot_id INT PRIMARY KEY,
  name VARCHAR(255),
  latitude NUMBER(10, 6),
  longitude NUMBER(10, 6)
);

CREATE TABLE ParkingSpace (
  parking_space_id INT PRIMARY KEY,
  parking_lot_id INT,
  space_number INT,
  status VARCHAR2(20),
  CONSTRAINT chk_status CHECK (status IN ('available', 'occupied', 'under_maintenance'))
);

CREATE TABLE Occupancy (
  occupancy_id INT PRIMARY KEY,
  parking_space_id INT,
  license_plate VARCHAR(20),
  start_time DATETIME,
  end_time DATETIME,
  duration INT,
  price DECIMAL(10,2)
);

CREATE TABLE StatusHistory (
  history_id INT PRIMARY KEY,
  parking_space_id INT,
  CONSTRAINT chk_status CHECK (status IN ('available', 'occupied', 'under_maintenance')),
  change_time DATETIME
);

CREATE TABLE Statistics (
  parking_lot_id INT,
  month DATE,
  completed_parkings INT
);

ALTER TABLE ParkingSpace ADD FOREIGN KEY (parking_lot_id) REFERENCES ParkingLot (parking_lot_id);

ALTER TABLE Occupancy ADD FOREIGN KEY (parking_space_id) REFERENCES ParkingSpace (parking_space_id);

ALTER TABLE StatusHistory ADD FOREIGN KEY (parking_space_id) REFERENCES ParkingSpace (parking_space_id);

ALTER TABLE Statistics ADD FOREIGN KEY (parking_lot_id) REFERENCES ParkingLot (parking_lot_id);
