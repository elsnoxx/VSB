from django import forms
from .models import Guest, Reservation, Payment, Employee, Room, RoomType, Address

class AddressForm(forms.ModelForm):
    class Meta:
        model = Address
        fields = ['street', 'city', 'postal_code', 'country']

class GuestForm(forms.ModelForm):
    class Meta:
        model = Guest
        fields = ['firstname', 'lastname', 'email', 'phone', 'birth_date', 'notes']
    
    # Ujisti se, že guest_type má výchozí hodnotu 'normal'
    guest_type = forms.CharField(widget=forms.HiddenInput(), initial='normal')

class ReservationForm(forms.ModelForm):
    class Meta:
        model = Reservation
        fields = ['guest', 'room', 'check_in_date', 'check_out_date', 'status']
        widgets = {
            'guest': forms.Select(attrs={'class': 'form-control'}),
            'room': forms.Select(attrs={'class': 'form-control'}),
            'check_in_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'check_out_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
        }

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