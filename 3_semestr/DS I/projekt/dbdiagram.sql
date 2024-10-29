Table Address {
  address_id int [pk, increment]
  street varchar
  city varchar
  postal_code char
  country varchar
}

Table Guest {
  guest_id int [pk, increment]
  firstname varchar
  lastname varchar
  email varchar
  phone varchar
  birth_date date
  address_id int [ref: > Address.address_id]
  guest_type varchar
  registration_date date
  notes text
}

Table Employee {
  employee_id int [pk, increment]
  firstname varchar
  lastname varchar
  position varchar
  address_id int [ref: > Address.address_id]
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
