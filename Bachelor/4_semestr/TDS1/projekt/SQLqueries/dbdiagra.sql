Table Guest {
  guest_id int [pk, increment]
  firstname varchar(100) [not null]
  lastname varchar(100) [not null]
  email varchar(100) [not null]
  phone varchar(30)
  birth_date date [not null]
  street varchar(255) [not null]
  city varchar(100) [not null]
  postal_code char(10) [not null]
  country varchar(100) [not null]
  guest_type varchar(50) [not null]
  registration_date date
  manager_id int
  notes varchar(2000)
}

Table Employee {
  employee_id int [pk, increment]
  firstname varchar(100) [not null]
  lastname varchar(100) [not null]
  position varchar(50) [not null]
  street varchar(255) [not null]
  city varchar(100) [not null]
  postal_code char(10) [not null]
  country varchar(100) [not null]
}

Table RoomType {
  room_type_id int [pk, increment]
  name varchar(100) [not null]
  bed_count int [not null]
}

Table Room {
  room_id int [pk, increment]
  room_type_id int [not null, ref: > RoomType.room_type_id]
  room_number varchar(20) [not null, unique]
  is_occupied boolean [not null, default: false]
}

Table Payment {
  payment_id int [pk, increment]
  total_accommodation decimal(10,2) [not null]
  total_expenses decimal(10,2) [not null]
  payment_date date
  is_paid boolean [not null, default: false]
}

Table Reservation {
  reservation_id int [pk, increment]
  guest_id int [not null, ref: > Guest.guest_id]
  room_id int [not null, ref: > Room.room_id]
  employee_id int [not null, ref: > Employee.employee_id]
  creation_date date [not null]
  check_in_date date [not null]
  check_out_date date [not null]
  payment_id int [not null, ref: > Payment.payment_id]
  status varchar(30) [not null]
  accommodation_price decimal(10,2)
}

Table Service {
  service_id int [pk, increment]
  name varchar(100) [not null]
  description varchar(2000)
}

Table ServiceUsage {
  usage_id int [pk, increment]
  reservation_id int [not null, ref: > Reservation.reservation_id]
  service_id int [not null, ref: > Service.service_id]
  quantity int [not null]
  total_price decimal(10,2) [not null]
}

Table Feedback {
  feedback_id int [pk, increment]
  guest_id int [not null, ref: > Guest.guest_id]
  reservation_id int [not null, ref: > Reservation.reservation_id]
  rating int [not null]
  note varchar(2000)
  feedback_date date
}

Table ServicePriceHistory {
  sph_id int [pk, increment]
  service_id int [not null, ref: > Service.service_id]
  price decimal(10,2) [not null]
  valid_from date [not null]
  valid_to date
}

Table RoomTypePriceHistory {
  rtph_id int [pk, increment]
  room_type_id int [not null, ref: > RoomType.room_type_id]
  price_per_night decimal(10,2) [not null]
  valid_from date [not null]
  valid_to date
}