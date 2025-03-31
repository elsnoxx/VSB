CREATE TABLE "Address" (
  "address_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "street" varchar,
  "city" varchar,
  "postal_code" char,
  "country" varchar
);

CREATE TABLE "Guest" (
  "guest_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "firstname" varchar,
  "lastname" varchar,
  "email" varchar,
  "phone" varchar,
  "birth_date" date,
  "address_id" int,
  "guest_type" varchar,
  "registration_date" date,
  "notes" text
);

CREATE TABLE "Employee" (
  "employee_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "firstname" varchar,
  "lastname" varchar,
  "position" varchar,
  "address_id" int
);

CREATE TABLE "RoomType" (
  "room_type_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "name" varchar,
  "bed_count" int,
  "price_per_night" decimal
);

CREATE TABLE "Room" (
  "room_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "room_type_id" int,
  "room_number" varchar,
  "is_occupied" boolean
);

CREATE TABLE "Payment" (
  "payment_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "total_accommodation" decimal,
  "total_expenses" decimal,
  "payment_date" date,
  "is_paid" boolean
);

CREATE TABLE "Reservation" (
  "reservation_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "guest_id" int,
  "room_id" int,
  "employee_id" int,
  "creation_date" date,
  "check_in_date" date,
  "check_out_date" date,
  "payment_id" int,
  "status" varchar
);

CREATE TABLE "Service" (
  "service_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "name" varchar,
  "description" text,
  "price" decimal
);

CREATE TABLE "ServiceUsage" (
  "usage_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "reservation_id" int,
  "service_id" int,
  "quantity" int,
  "total_price" decimal
);

CREATE TABLE "Feedback" (
  "feedback_id" int GENERATED AS IDENTITY PRIMARY KEY,
  "guest_id" int,
  "reservation_id" int,
  "rating" int,
  "comment" text,
  "feedback_date" date
);

ALTER TABLE "Guest" ADD FOREIGN KEY ("address_id") REFERENCES "Address" ("address_id");

ALTER TABLE "Employee" ADD FOREIGN KEY ("address_id") REFERENCES "Address" ("address_id");

ALTER TABLE "Room" ADD FOREIGN KEY ("room_type_id") REFERENCES "RoomType" ("room_type_id");

ALTER TABLE "Reservation" ADD FOREIGN KEY ("guest_id") REFERENCES "Guest" ("guest_id");

ALTER TABLE "Reservation" ADD FOREIGN KEY ("room_id") REFERENCES "Room" ("room_id");

ALTER TABLE "Reservation" ADD FOREIGN KEY ("employee_id") REFERENCES "Employee" ("employee_id");

ALTER TABLE "Reservation" ADD FOREIGN KEY ("payment_id") REFERENCES "Payment" ("payment_id");

ALTER TABLE "ServiceUsage" ADD FOREIGN KEY ("reservation_id") REFERENCES "Reservation" ("reservation_id");

ALTER TABLE "ServiceUsage" ADD FOREIGN KEY ("service_id") REFERENCES "Service" ("service_id");

ALTER TABLE "Feedback" ADD FOREIGN KEY ("guest_id") REFERENCES "Guest" ("guest_id");

ALTER TABLE "Feedback" ADD FOREIGN KEY ("reservation_id") REFERENCES "Reservation" ("reservation_id");
