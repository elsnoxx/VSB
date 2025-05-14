from datetime import date
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.contrib import messages
from .models import Guest, Reservation, Room, Employee, Payment
from .forms import GuestForm, ReservationForm, RoomForm, EmployeeForm, CustomUserCreationForm, GuestRegisterForm
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, UserChangeForm
from django.contrib.auth.decorators import login_required
import json


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

def register_view(request):
    if request.method == 'POST':
        form = GuestRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email']
            user.first_name = form.cleaned_data['firstname']
            user.last_name = form.cleaned_data['lastname']
            user.save()
            Guest.objects.create(
                user=user,
                firstname=form.cleaned_data['firstname'],
                lastname=form.cleaned_data['lastname'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data['phone'],
                birth_date=form.cleaned_data['birth_date'],
                street=form.cleaned_data['street'],
                city=form.cleaned_data['city'],
                postal_code=form.cleaned_data['postal_code'],
                country=form.cleaned_data['country'],
                guest_type='normal',
                notes=form.cleaned_data['notes']
            )
            messages.success(request, 'Registrace proběhla úspěšně. Nyní se můžete přihlásit.')
            return redirect('login')
    else:
        form = GuestRegisterForm()
    return render(request, 'register.html', {'form': form})

@login_required
def profile_view(request):
    return render(request, 'profile.html', {'user': request.user})

@login_required
def profile_edit(request):
    if request.method == 'POST':
        form = UserChangeForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profil byl upraven.')
            return redirect('profile')
    else:
        form = UserChangeForm(instance=request.user)
    return render(request, 'profile_edit.html', {'form': form})

# Hlavní stránka
def index(request):
    rooms = Room.objects.all()
    
    floors = {
        "1": [],
        "2": [],
        "3": []
    }

    for room in rooms:
        floor_number = str(room.room_number)[0]
        if floor_number in floors:
            floors[floor_number].append(room)
    
    return render(request, 'management/reservation/reservation_rooms.html', {'floors': floors})

@login_required
def create_reservation(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    guest, created = Guest.objects.get_or_create(
        user=request.user,
        defaults={
            "firstname": request.user.first_name or "Neznámý",
            "lastname": request.user.last_name or "Uživatel",
            "email": request.user.email or "",
            "birth_date": "2000-01-01",
            "street": "",
            "city": "",
            "postal_code": "",
            "country": "",
            "guest_type": "normal"
        }
    )
    # Výchozí datumy (např. dnes a zítra)
    from datetime import date, timedelta
    default_check_in = date.today()
    default_check_out = date.today() + timedelta(days=1)
    price = room.room_type.price_per_night

    if request.method == 'POST':
        form = ReservationForm(request.POST)
        if form.is_valid():
            reservation = form.save(commit=False)
            reservation.guest = guest
            reservation.room = room
            reservation.accommodation_price = price
            reservation.status = "Nová"
            # případně nastav další pole (employee, payment, ...)
            reservation.save()
            return redirect('reservation_list')
    else:
        form = ReservationForm(initial={
            'check_in_date': default_check_in,
            'check_out_date': default_check_out,
        })
    return render(request, 'create_reservation.html', {
        'form': form,
        'room': room,
        'guest': guest,
        'price': price,
        'default_check_in': default_check_in,
        'default_check_out': default_check_out,
    })

@login_required
def guest_list(request):
    guests = Guest.objects.all()
    return render(request, 'management/guest/guest_list.html', {'guests': guests})

# Detail hosta
def guest_detail(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    return render(request, 'management/guest/guest_detail.html', {'guest': guest})

@login_required
# Vytvoření hosta
def guest_create(request):
    if request.method == 'POST':
        guest_form = GuestForm(request.POST)
        if guest_form.is_valid():
            guest = guest_form.save(commit=False)
            guest.guest_type = 'normal'  # nastav výchozí typ hosta
            guest.save()
            return redirect('guest_list')
        else:
            return render(request, 'management/guest/guest_form.html', {'form': guest_form})
    else:
        guest_form = GuestForm()
    return render(request, 'management/guest/guest_form.html', {'form': guest_form})

@login_required
def add_guest(request):
    if request.method == 'POST':
        guest_form = GuestForm(request.POST)
        if guest_form.is_valid():
            guest = guest_form.save(commit=False)
            guest.guest_type = 'normal'
            guest.save()
            return redirect('guest_list')
    else:
        guest_form = GuestForm()
    return render(request, 'management/guest/guest_form.html', {'form': guest_form})

@login_required
# Room ocupancies
def room_occupancies(request):
    return render(request, 'room_occupancies.html')

@login_required
# Editace hosta
def guest_update(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    if request.method == "POST":
        form = GuestForm(request.POST, instance=guest)
        if form.is_valid():
            form.save()
            return redirect('guest_list')
    else:
        form = GuestForm(instance=guest)
    return render(request, 'management/guest/guest_form.html', {'form': form})


@login_required
# Smazání hosta
def guest_delete(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    if request.method == "POST":
        guest.delete()
        return JsonResponse({'success': True})  # Vrátí JSON odpověď
    return JsonResponse({'success': False}, status=400)


@login_required
# Seznam rezervací
def reservation_list(request):
    reservations = Reservation.objects.all()
    return render(request, 'management/reservation/reservation_list.html', {'reservations': reservations})


@login_required
# Detail rezervace
def reservation_detail(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    guest = reservation.guest
    room = reservation.room
    payment = reservation.payment
    context = {
        'reservation': reservation,
        'guest': guest,
        'room': room,
        'payment': payment,
    }
    return render(request, 'management/reservation/reservation_detail.html', context)


@login_required
def mark_payment_as_paid_from_reservation(request, reservation_id):
    if request.method == "POST":
        reservation = get_object_or_404(Reservation, pk=reservation_id)
        payment = reservation.payment
        if payment:
            payment.is_paid = True
            payment.payment_date = date.today()
            payment.save()
        return redirect('reservation_detail', reservation_id=reservation_id)
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@login_required
def reservation_update(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    if request.method == "POST":
        form = ReservationForm(request.POST, instance=reservation)
        if form.is_valid():
            form.save()
            return redirect('reservation_list')
    else:
        form = ReservationForm(instance=reservation)
    return render(request, 'management/reservation/reservation_form.html', {'form': form})


@login_required
def reservation_delete(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    payment = reservation.payment
    

    if request.method == "POST":
        reservation.delete()
        payment.delete() if payment else None
        return redirect('reservation_list')
    return render(request, 'management/reservation/reservation_confirm_delete.html', {'reservation': reservation})


@login_required
def room_management(request):
    # Získání všech místností a jejich typů
    rooms = Room.objects.select_related('room_type').all()
    return render(request, 'management/room/room_management.html', {'rooms': rooms})


@login_required
def room_create(request):
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('room_management')
    else:
        form = RoomForm()
    return render(request, 'forms/room_form.html', {'form': form})

@login_required
def room_update(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    if request.method == "POST":
        form = RoomForm(request.POST, instance=room)
        if form.is_valid():
            form.save()
            return redirect('room_management')
    else:
        form = RoomForm(instance=room)
    return render(request, 'forms/room_form.html', {'form': form})


@csrf_exempt
@login_required
def room_delete(request, room_id):
    if request.method == "POST":
        room = get_object_or_404(Room, pk=room_id)
        room.delete()
        return redirect('room_management') 

@login_required
# Rezerzervace 
def reservation_success(request, roomid):
    room = get_object_or_404(Room, pk=roomid)
    if room:
        room.is_occupied = True
        room.save()
        return render(request, 'reservation_success.html', {'room': room})

@login_required
def employe_management(request):
    emp = Employee.objects.all()
    return render(request, 'management/employe/employe_management.html', {'emp': emp})

@login_required
def employe_create(request):
    if request.method == 'POST':
        form = EmployeeForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('employe_management')  # Přesměrování na stránku správy zaměstnanců
    else:
        form = EmployeeForm()
    return render(request, 'forms/employe_form.html', {'form': form})

@login_required
def employe_update(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    if request.method == "POST":
        form = EmployeeForm(request.POST, instance=employee)
        if form.is_valid():
            form.save()
            return redirect('employe_management')
    else:
        form = EmployeeForm(instance=employee)
    return render(request, 'forms/employee_form.html', {'form': form})

@login_required
def employe_delete(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    if request.method == "POST":
        employee.delete()
        return redirect('employe_management')
    return render(request, 'management/employe/employe_confirm_delete.html', {'employee': employee})

@login_required
def payment_management(request):
    payment = Payment.objects.all()
    return render(request, 'management/payment/payment_management.html', {'payment': payment})

@login_required
@csrf_exempt
def mark_payment_as_paid(request, employee_id):
    if request.method == "POST":
        payment = get_object_or_404(Payment, pk=employee_id)
        payment.is_paid = True
        payment.payment_date = date.today()
        payment.save()
        return redirect('payment_management')
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@login_required
def mark_payment_as_paid(request, employee_id):
    if request.method == "POST":
        payment = get_object_or_404(Payment, pk=employee_id)
        payment.is_paid = True
        payment.save()
        return redirect('payment_management')
    return JsonResponse({'error': 'Invalid request method'}, status=400)

