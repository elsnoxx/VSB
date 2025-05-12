-- Nejprve dropni tabulky, které mají cizí klíče na jiné tabulky
DROP TABLE IF EXISTS Feedback;
DROP TABLE IF EXISTS ServiceUsage;
DROP TABLE IF EXISTS Reservation;
DROP TABLE IF EXISTS ServicePriceHistory;
DROP TABLE IF EXISTS RoomTypePriceHistory;
DROP TABLE IF EXISTS Payment;
DROP TABLE IF EXISTS Service;
DROP TABLE IF EXISTS Room;
DROP TABLE IF EXISTS RoomType;
DROP TABLE IF EXISTS Employee;
DROP TABLE IF EXISTS Guest;

-- Guest table
CREATE TABLE Guest (
  guest_id INT IDENTITY(1,1) PRIMARY KEY,
  firstname NVARCHAR(100) NOT NULL,
  lastname NVARCHAR(100) NOT NULL,
  email NVARCHAR(100) NOT NULL,
  phone NVARCHAR(30),
  birth_date DATE NOT NULL,
  street NVARCHAR(255) NOT NULL,
  city NVARCHAR(100) NOT NULL,
  postal_code CHAR(10) NOT NULL,
  country NVARCHAR(100) NOT NULL,
  guest_type NVARCHAR(50) NOT NULL,
  registration_date DATE DEFAULT GETDATE(),
  notes NVARCHAR(MAX)
);

-- Employee table
CREATE TABLE Employee (
  employee_id INT IDENTITY(1,1) PRIMARY KEY,
  firstname NVARCHAR(100) NOT NULL,
  lastname NVARCHAR(100) NOT NULL,
  position NVARCHAR(50) NOT NULL,
  street NVARCHAR(255) NOT NULL,
  city NVARCHAR(100) NOT NULL,
  postal_code CHAR(10) NOT NULL,
  country NVARCHAR(100) NOT NULL
);

-- RoomType table
CREATE TABLE RoomType (
  room_type_id INT IDENTITY(1,1) PRIMARY KEY,
  name NVARCHAR(100) NOT NULL,
  bed_count INT NOT NULL
);

-- Room table
CREATE TABLE Room (
  room_id INT IDENTITY(1,1) PRIMARY KEY,
  room_type_id INT NOT NULL,
  room_number NVARCHAR(20) NOT NULL,
  is_occupied BIT DEFAULT 0 NOT NULL,
  CONSTRAINT fk_room_roomtype FOREIGN KEY (room_type_id) REFERENCES RoomType(room_type_id),
  CONSTRAINT uq_room_number UNIQUE (room_number)
);

-- Payment table
CREATE TABLE Payment (
  payment_id INT IDENTITY(1,1) PRIMARY KEY,
  total_accommodation DECIMAL(10,2) NOT NULL,
  total_expenses DECIMAL(10,2) NOT NULL,
  payment_date DATE,
  is_paid BIT DEFAULT 0 NOT NULL
);

-- Reservation table
CREATE TABLE Reservation (
  reservation_id INT IDENTITY(1,1) PRIMARY KEY,
  guest_id INT NOT NULL,
  room_id INT NOT NULL,
  employee_id INT NOT NULL,
  creation_date DATE DEFAULT GETDATE() NOT NULL,
  check_in_date DATE NOT NULL,
  check_out_date DATE NOT NULL,
  payment_id INT NOT NULL,
  status NVARCHAR(30) NOT NULL,
  accommodation_price DECIMAL(10,2),
  CONSTRAINT fk_reservation_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_reservation_room FOREIGN KEY (room_id) REFERENCES Room(room_id),
  CONSTRAINT fk_reservation_employee FOREIGN KEY (employee_id) REFERENCES Employee(employee_id),
  CONSTRAINT fk_reservation_payment FOREIGN KEY (payment_id) REFERENCES Payment(payment_id)
);

-- Service table
CREATE TABLE Service (
  service_id INT IDENTITY(1,1) PRIMARY KEY,
  name NVARCHAR(100) NOT NULL,
  description NVARCHAR(MAX)
);

-- ServiceUsage table
CREATE TABLE ServiceUsage (
  usage_id INT IDENTITY(1,1) PRIMARY KEY,
  reservation_id INT NOT NULL,
  service_id INT NOT NULL,
  quantity INT NOT NULL,
  total_price DECIMAL(10,2) NOT NULL,
  CONSTRAINT fk_serviceusage_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id),
  CONSTRAINT fk_serviceusage_service FOREIGN KEY (service_id) REFERENCES Service(service_id)
);

-- Feedback table
CREATE TABLE Feedback (
  feedback_id INT IDENTITY(1,1) PRIMARY KEY,
  guest_id INT NOT NULL,
  reservation_id INT NOT NULL,
  rating INT NOT NULL,
  comment NVARCHAR(MAX),
  feedback_date DATE DEFAULT GETDATE() NOT NULL,
  CONSTRAINT fk_feedback_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_feedback_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id)
);

-- ServicePriceHistory table
CREATE TABLE ServicePriceHistory (
  sph_id INT IDENTITY(1,1) PRIMARY KEY,
  service_id INT NOT NULL,
  price DECIMAL(10,2) NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  CONSTRAINT fk_sph_service FOREIGN KEY (service_id) REFERENCES Service(service_id)
);

-- RoomTypePriceHistory table
CREATE TABLE RoomTypePriceHistory (
  rtph_id INT IDENTITY(1,1) PRIMARY KEY,
  room_type_id INT NOT NULL,
  price_per_night DECIMAL(10,2) NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  CONSTRAINT fk_rtph_roomtype FOREIGN KEY (room_type_id) REFERENCES RoomType(room_type_id)
);