from django import forms
from .models import Guest, Reservation, Payment, Employee, Room, RoomType, Service, Feedback
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class GuestForm(forms.ModelForm):
    class Meta:
        model = Guest
        fields = '__all__'
        widgets = {
            'firstname': forms.TextInput(attrs={'class': 'form-control'}),
            'lastname': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'birth_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        }

    # Ujisti se, že guest_type má výchozí hodnotu 'normal'
    guest_type = forms.CharField(widget=forms.HiddenInput(), initial='normal')

class ReservationForm(forms.ModelForm):
    class Meta:
        model = Reservation
        exclude = ['guest', 'room', 'accommodation_price', 'status', 'employee', 'payment']
        widgets = {
            'check_in_date': forms.DateInput(attrs={'type': 'date'}),
            'check_out_date': forms.DateInput(attrs={'type': 'date'}),
        }

class PaymentForm(forms.ModelForm):
    class Meta:
        model = Payment
        fields = ['total_accommodation', 'total_expenses', 'payment_date', 'is_paid']

class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = ['firstname', 'lastname', 'position', 'street', 'city', 'postal_code', 'country']

class RoomForm(forms.ModelForm):
    class Meta:
        model = Room
        fields = ['room_type', 'room_number', 'is_occupied']

class RoomTypeForm(forms.ModelForm):
    class Meta:
        model = RoomType
        fields = ['name', 'bed_count']

class RoomForm(forms.ModelForm):
    room_type_details = forms.CharField(
        label="Detaily typu pokoje",
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'readonly': 'readonly', 'rows': 3}),
    )

    class Meta:
        model = Room
        fields = ['room_number', 'room_type', 'is_occupied']
        widgets = {
            'room_number': forms.TextInput(attrs={'class': 'form-control'}),
            'room_type': forms.Select(attrs={'class': 'form-control'}),
            'is_occupied': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and hasattr(self.instance, 'room_type') and self.instance.room_type:
            room_type = self.instance.room_type
            self.fields['room_type_details'].initial = (
                f"Název: {room_type.name}\n"
                f"Počet postelí: {room_type.bed_count}\n"
                f"Cena za noc: {room_type.price_per_night} Kč"
            )
        else:
            self.fields['room_type_details'].initial = "Žádný typ pokoje není přiřazen."


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(label="Jméno", max_length=30, required=False)
    last_name = forms.CharField(label="Příjmení", max_length=30, required=False)

    class Meta:
        model = User
        fields = ("username", "email", "first_name", "last_name", "password1", "password2")

class GuestRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    firstname = forms.CharField(max_length=100)
    lastname = forms.CharField(max_length=100)
    phone = forms.CharField(max_length=30, required=False)
    birth_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    street = forms.CharField(max_length=255)
    city = forms.CharField(max_length=100)
    postal_code = forms.CharField(max_length=10)
    country = forms.CharField(max_length=100)
    notes = forms.CharField(widget=forms.Textarea, required=False)

    class Meta:
        model = User
        fields = ("username", "password1", "password2", "email", "firstname", "lastname", "phone", "birth_date", "street", "city", "postal_code", "country", "notes")

class ServiceForm(forms.ModelForm):
    class Meta:
        model = Service
        fields = ['name', 'description', 'price']
        
class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['rating', 'comment']
        widgets = {
            'rating': forms.NumberInput(attrs={'min': 1, 'max': 5, 'class': 'form-control'}),
            'comment': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': ' '}),
        label='Uživatelské jméno'
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': ' '}),
        label='Heslo'
    )