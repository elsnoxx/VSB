from django import forms
from .models import Guest, Reservation, Payment, Employee, Room, RoomType

class GuestForm(forms.ModelForm):
    class Meta:
        model = Guest
        fields = ['firstname', 'lastname', 'email', 'phone', 'birth_date', 'guest_type', 'notes']

class ReservationForm(forms.ModelForm):
    class Meta:
        model = Reservation
        fields = ['guest', 'room', 'employee', 'check_in_date', 'check_out_date', 'status']

class PaymentForm(forms.ModelForm):
    class Meta:
        model = Payment
        fields = ['total_accommodation', 'total_expenses', 'payment_date', 'is_paid']

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['firstname', 'lastname', 'position', 'address']

class RoomForm(forms.ModelForm):
    class Meta:
        model = Room
        fields = ['room_type', 'room_number', 'is_occupied']

class RoomTypeForm(forms.ModelForm):
    class Meta:
        model = RoomType
        fields = ['name', 'bed_count', 'price_per_night']
