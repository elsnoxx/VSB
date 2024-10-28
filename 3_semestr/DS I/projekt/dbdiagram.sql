// https://dbdiagram.io/d

Table Guest {
  guest_id int [pk, increment] // Primární klíč
  name varchar(100)
  email varchar(100)
  phone varchar(15)
}

Table RoomType {
  room_type_id int [pk, increment] // Primární klíč
  name varchar(50) // Název typu pokoje
  bed_count int // Počet postelí
  price_per_night decimal(10, 2) // Cena za noc
}

Table Room {
  room_id int [pk, increment] // Primární klíč
  room_type_id int [ref: > RoomType.room_type_id] // Cizí klíč na typ pokoje
  room_number varchar(10) // Číslo pokoje
  is_occupied boolean // Zda je pokoj obsazený
}

Table Reservation {
  reservation_id int [pk, increment] // Primární klíč
  guest_id int [ref: > Guest.guest_id] // Cizí klíč na hosta
  room_id int [ref: > Room.room_id] // Cizí klíč na pokoj
  employee_id int [ref: > Employee.employee_id] // Cizí klíč na zaměstnance
  creation_date date // Datum vytvoření rezervace
  check_in_date date // Datum příjezdu
  check_out_date date // Datum odjezdu
  status varchar(20) // Stav rezervace (např. aktivní, zrušená)
}

Table Employee {
  employee_id int [pk, increment] // Primární klíč
  name varchar(100) // Jméno zaměstnance
  position varchar(50) // Pozice zaměstnance
}

Table Expense {
  expense_id int [pk, increment] // Primární klíč
  reservation_id int [ref: > Reservation.reservation_id] // Cizí klíč na rezervaci
  description varchar(255) // Popis útraty
  amount decimal(10, 2) // Částka za službu nebo útratu
  days int // Počet dnů, po které je útrata účtována
}

Table Payment {
  payment_id int [pk, increment] // Primární klíč
  reservation_id int [ref: > Reservation.reservation_id] // Cizí klíč na rezervaci
  total_accommodation decimal(10, 2) // Celková částka za ubytování
  total_expenses decimal(10, 2) // Celková částka za útraty
  payment_date date // Datum platby
  is_paid boolean // Zda byla platba provedena
}
