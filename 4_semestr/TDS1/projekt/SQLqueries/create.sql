-- Guest table
CREATE TABLE Guest (
  guest_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  email VARCHAR2(100) NOT NULL,
  phone VARCHAR2(30),
  birth_date DATE NOT NULL,
  street VARCHAR2(255) NOT NULL,
  city VARCHAR2(100) NOT NULL,
  postal_code CHAR(10) NOT NULL,
  country VARCHAR2(100) NOT NULL,
  guest_type VARCHAR2(50) NOT NULL,
  registration_date DATE DEFAULT SYSDATE,
  notes VARCHAR2(2000)
);

-- Employee table
CREATE TABLE Employee (
  employee_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  firstname VARCHAR2(100) NOT NULL,
  lastname VARCHAR2(100) NOT NULL,
  position VARCHAR2(50) NOT NULL,
  street VARCHAR2(255) NOT NULL,
  city VARCHAR2(100) NOT NULL,
  postal_code CHAR(10) NOT NULL,
  country VARCHAR2(100) NOT NULL
);

-- RoomType table
CREATE TABLE RoomType (
  room_type_id NUMBER PRIMARY KEY,
  name VARCHAR2(100) NOT NULL,
  bed_count NUMBER NOT NULL
);

-- Room table
CREATE TABLE Room (
  room_id NUMBER PRIMARY KEY,
  room_type_id NUMBER NOT NULL,
  room_number VARCHAR2(20) NOT NULL,
  is_occupied NUMBER(1) DEFAULT 0 NOT NULL,
  CONSTRAINT fk_room_roomtype FOREIGN KEY (room_type_id) REFERENCES RoomType(room_type_id),
  CONSTRAINT uq_room_number UNIQUE (room_number)
);

-- Payment table
CREATE TABLE Payment (
  payment_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  total_accommodation NUMBER(10,2) NOT NULL,
  total_expenses NUMBER(10,2) NOT NULL,
  payment_date DATE,
  is_paid NUMBER(1) DEFAULT 0 NOT NULL
);

-- Reservation table
CREATE TABLE Reservation (
  reservation_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  room_id NUMBER NOT NULL,
  employee_id NUMBER NOT NULL,
  creation_date DATE DEFAULT SYSDATE NOT NULL,
  check_in_date DATE NOT NULL,
  check_out_date DATE NOT NULL,
  payment_id NUMBER NOT NULL,
  status VARCHAR2(30) NOT NULL,
  accommodation_price NUMBER(10,2),
  CONSTRAINT fk_reservation_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_reservation_room FOREIGN KEY (room_id) REFERENCES Room(room_id),
  CONSTRAINT fk_reservation_employee FOREIGN KEY (employee_id) REFERENCES Employee(employee_id),
  CONSTRAINT fk_reservation_payment FOREIGN KEY (payment_id) REFERENCES Payment(payment_id)
);

-- Service table
CREATE TABLE Service (
  service_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR2(100) NOT NULL,
  description VARCHAR2(2000)
);

-- ServiceUsage table
CREATE TABLE ServiceUsage (
  usage_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  reservation_id NUMBER NOT NULL,
  service_id NUMBER NOT NULL,
  quantity NUMBER NOT NULL,
  total_price NUMBER(10,2) NOT NULL,
  CONSTRAINT fk_serviceusage_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id),
  CONSTRAINT fk_serviceusage_service FOREIGN KEY (service_id) REFERENCES Service(service_id)
);

-- Feedback table
CREATE TABLE Feedback (
  feedback_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  guest_id NUMBER NOT NULL,
  reservation_id NUMBER NOT NULL,
  rating NUMBER NOT NULL,
  note VARCHAR2(2000),
  feedback_date DATE DEFAULT SYSDATE NOT NULL,
  CONSTRAINT fk_feedback_guest FOREIGN KEY (guest_id) REFERENCES Guest(guest_id),
  CONSTRAINT fk_feedback_reservation FOREIGN KEY (reservation_id) REFERENCES Reservation(reservation_id)
);

-- ServicePriceHistory table
CREATE TABLE ServicePriceHistory (
  sph_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  service_id NUMBER NOT NULL,
  price NUMBER(10,2) NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  CONSTRAINT fk_sph_service FOREIGN KEY (service_id) REFERENCES Service(service_id)
);

-- RoomTypePriceHistory table
CREATE TABLE RoomTypePriceHistory (
  rtph_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  room_type_id NUMBER NOT NULL,
  price_per_night NUMBER(10,2) NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  CONSTRAINT fk_rtph_roomtype FOREIGN KEY (room_type_id) REFERENCES RoomType(room_type_id)
);