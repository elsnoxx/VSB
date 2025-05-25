from datetime import date
from decimal import Decimal
from ninja import Schema
from pydantic import Field
from typing import Optional

class GuestSchema(Schema):
    guest_id: int
    firstname: str
    lastname: str
    email: str
    phone: Optional[str]
    birth_date: date
    street: str
    city: str
    postal_code: str
    country: str
    guest_type: str
    registration_date: date
    notes: Optional[str]

class EmployeeSchema(Schema):
    employee_id: int
    firstname: str
    lastname: str
    position: str
    street: str
    city: str
    postal_code: str
    country: str

class RoomTypeSchema(Schema):
    room_type_id: int
    name: str
    bed_count: int
    price_per_night: Decimal
    description: Optional[str]

class RoomSchema(Schema):
    room_id: int
    room_type_id: int
    room_number: str
    is_occupied: bool

class PaymentSchema(Schema):
    payment_id: int
    total_accommodation: Decimal
    total_expenses: Decimal
    payment_date: Optional[date]
    is_paid: bool

class ReservationSchema(Schema):
    reservation_id: int
    guest: int = Field(..., alias="guest_id")
    room: int = Field(..., alias="room_id")
    creation_date: date
    check_in_date: date
    check_out_date: date
    payment: int = Field(..., alias="payment_id")
    status: str
    accommodation_price: Optional[Decimal]

class ServiceSchema(Schema):
    service_id: int
    name: str
    description: Optional[str]
    price: Decimal

class ServiceUsageSchema(Schema):
    usage_id: int
    reservation: int = Field(..., alias="reservation_id")
    service: int = Field(..., alias="service_id")
    quantity: int
    total_price: Decimal

class FeedbackSchema(Schema):
    feedback_id: int
    guest: int = Field(..., alias="guest_id")
    reservation: int = Field(..., alias="reservation_id")
    rating: int
    comment: Optional[str]
    feedback_date: date

class AddressSchema(Schema):
    address_id: int
    street: str
    city: str
    postal_code: str
    country: str