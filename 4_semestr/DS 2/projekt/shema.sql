Table Guest {
  guest_id int [pk, increment]
  firstname varchar
  lastname varchar
  email varchar
  phone varchar
  birth_date date
  street varchar
  city varchar
  postal_code char
  country varchar
  guest_type varchar
  registration_date date
  notes text
}

Table Employee {
  employee_id int [pk, increment]
  firstname varchar
  lastname varchar
  position varchar
  street varchar
  city varchar
  postal_code char
  country varchar
}

Table RoomType {
  room_type_id int [pk, increment]
  name varchar
  bed_count int
  price_per_night decimal
}

Table Room {
  room_id int [pk, increment]
  room_type_id int [ref: > RoomType.room_type_id]
  room_number varchar
  is_occupied boolean
}

Table Payment {
  payment_id int [pk, increment]
  total_accommodation decimal
  total_expenses decimal
  payment_date date
  is_paid boolean
}

Table Reservation {
  reservation_id int [pk, increment]
  guest_id int [ref: > Guest.guest_id]
  room_id int [ref: > Room.room_id]
  employee_id int [ref: > Employee.employee_id]
  creation_date date
  check_in_date date
  check_out_date date
  payment_id int [ref: > Payment.payment_id]
  status varchar
}

Table Service {
  service_id int [pk, increment]
  name varchar
  description text
  price decimal
}

Table ServiceUsage {
  usage_id int [pk, increment]
  reservation_id int [ref: > Reservation.reservation_id]
  service_id int [ref: > Service.service_id]
  quantity int
  total_price decimal
}

Table Feedback {
  feedback_id int [pk, increment]
  guest_id int [ref: > Guest.guest_id]
  reservation_id int [ref: > Reservation.reservation_id]
  rating int
  comment text
  feedback_date date
}

Table ServicePriceHistory {
  sph_id int [pk, increment]
  service_id int [ref: > Service.service_id]
  price decimal
  valid_from date
  valid_to date
}

Table RoomTypePriceHistory {
  rtph_id int [pk, increment]
  room_type_id int [ref: > RoomType.room_type_id]
  price_per_night decimal
  valid_from date
  valid_to date
}
