from ninja import NinjaAPI
from .models import Guest, Employee, RoomType, Room, Reservation, Service, Payment, ServiceUsage, Feedback, Address
from .schemas import GuestSchema, EmployeeSchema, RoomTypeSchema, RoomSchema, ReservationSchema, ServiceSchema, PaymentSchema, ServiceUsageSchema, FeedbackSchema, AddressSchema
from typing import List

api = NinjaAPI()

@api.get("/guests/", response=List[GuestSchema])
def list_guests(request):
    return Guest.objects.all()

@api.get("/employees/", response=List[EmployeeSchema])
def list_employees(request):
    return Employee.objects.all()

@api.get("/roomtypes/", response=List[RoomTypeSchema])
def list_roomtypes(request):
    return RoomType.objects.all()

@api.get("/rooms/", response=List[RoomSchema])
def list_rooms(request):
    return Room.objects.all()

@api.get("/reservations/", response=List[ReservationSchema])
def list_reservations(request):
    return Reservation.objects.values(
        "reservation_id",
        "guest_id",
        "room_id",
        "creation_date",
        "check_in_date",
        "check_out_date",
        "payment_id",
        "status",
        "accommodation_price"
    )

@api.get("/services/", response=List[ServiceSchema])
def list_services(request):
    return Service.objects.all()

@api.get("/payments/", response=List[PaymentSchema])
def list_payments(request):
    return Payment.objects.all()

@api.get("/serviceusages/", response=List[ServiceUsageSchema])
def list_serviceusages(request):
    return ServiceUsage.objects.values(
        "usage_id",
        "reservation_id",
        "service_id",
        "quantity",
        "total_price"
    )

@api.get("/feedbacks/", response=List[FeedbackSchema])
def list_feedbacks(request):
    return Feedback.objects.values(
        "feedback_id",
        "guest_id",
        "reservation_id",
        "rating",
        "comment",
        "feedback_date"
    )

@api.get("/addresses/", response=List[AddressSchema])
def list_addresses(request):
    return Address.objects.all()