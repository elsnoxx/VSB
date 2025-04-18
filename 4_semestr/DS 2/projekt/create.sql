-- Create Address table
CREATE TABLE Address (
  address_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  street VARCHAR2(255) NOT NULL,
  city VARCHAR2(100) NOT NULL,
  postal_code CHAR(10) NOT NULL,
  country VARCHAR2(100) NOT NULL
);

-- Create Guest table
CREATE TABLE Guest (
  guest_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  email VARCHAR2(100) NOT NULL,
  phone VARCHAR2(15) NULL,
  birth_date DATE NOT NULL,
  address_id NUMBER NOT NULL,
  guest_type VARCHAR2(50) NOT NULL,
  registration_date DATE DEFAULT SYSDATE,
  notes CLOB,
  CONSTRAINT fk_guest_address FOREIGN KEY (address_id) REFERENCES Address(address_id)
);

-- Create Employee table
CREATE TABLE Employee (
  employee_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  position VARCHAR2(50) NOT NULL,
  address_id NUMBER NOT NULL,
  CONSTRAINT fk_employee_address FOREIGN KEY (address_id) REFERENCES Address(address_id)
);

-- Create RoomType table
CREATE TABLE RoomType (
  room_type_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR2(50) NOT NULL,
  bed_count NUMBER NOT NULL,
  price_per_night NUMBER(10,2) NOT NULL
);

-- Create Room table
CREATE TABLE Room (
  room_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  room_type_id NUMBER NOT NULL,
  room_number VARCHAR2(10) NOT NULL,
  is_occupied NUMBER(1) NOT NULL,
  CONSTRAINT fk_room_roomtype FOREIGN KEY (room_type_id) REFERENCES RoomType(room_type_id),
  CONSTRAINT uq_room_number UNIQUE (room_number)
);

-- Create Payment table
CREATE TABLE Payment (
  payment_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  total_accommodation NUMBER(10,2) NOT NULL,
  total_expenses NUMBER(10,2) NOT NULL,
  payment_date DATE NOT NULL,
  is_paid NUMBER(1) NOT NULL
);

-- Create Reservation table
CREATE TABLE Reservation (
  reservation_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  room_id NUMBER NOT NULL,
  employee_id NUMBER NOT NULL,
  creation_date DATE NOT NULL,
  check_in_date DATE NOT NULL,
  check_out_date DATE NOT NULL,
  payment_id NUMBER NOT NULL,
  status VARCHAR2(20) NOT NULL,
  CONSTRAINT fk_reservation_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_reservation_room FOREIGN KEY (room_id) REFERENCES Room(room_id),
  CONSTRAINT fk_reservation_employee FOREIGN KEY (employee_id) REFERENCES Employee(employee_id),
  CONSTRAINT fk_reservation_payment FOREIGN KEY (payment_id) REFERENCES Payment(payment_id),
  CONSTRAINT ck_reservation_status CHECK (status IN ('Confirmed', 'Checked In', 'Checked Out', 'Cancelled')),
  CONSTRAINT ck_check_dates CHECK (check_in_date < check_out_date)
);

-- Create Service table
CREATE TABLE Service (
  service_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR2(100) NOT NULL,
  description CLOB,
  price NUMBER(10,2) NOT NULL,
  CONSTRAINT ck_service_price CHECK (price >= 0)
);

-- Create ServiceUsage table
CREATE TABLE ServiceUsage (
  usage_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  reservation_id NUMBER NOT NULL,
  service_id NUMBER NOT NULL,
  quantity NUMBER NOT NULL,
  usage_date DATE DEFAULT SYSDATE NOT NULL,
  total_price NUMBER(10,2) NOT NULL,
  CONSTRAINT fk_serviceusage_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id),
  CONSTRAINT fk_serviceusage_service FOREIGN KEY (service_id) REFERENCES Service(service_id),
  CONSTRAINT ck_serviceusage_quantity CHECK (quantity > 0),
  CONSTRAINT ck_serviceusage_price CHECK (total_price >= 0)
);

-- Create Feedback table
CREATE TABLE Feedback (
  feedback_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  reservation_id NUMBER NOT NULL,
  rating NUMBER NOT NULL,
  comment CLOB,
  feedback_date DATE DEFAULT SYSDATE NOT NULL,
  CONSTRAINT fk_feedback_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_feedback_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id),
  CONSTRAINT ck_feedback_rating CHECK (rating BETWEEN 1 AND 5)
);

-- Přidání integritního omezení pro guest_type
ALTER TABLE Guest
  ADD CONSTRAINT ck_guest_type CHECK (guest_type IN ('Regular', 'VIP', 'Loyalty'));
