from datetime import date
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from .models import Guest, Reservation, Room, Employee, Payment
from .forms import GuestForm, ReservationForm, AddressForm, RoomForm
from django.views.decorators.csrf import csrf_exempt
import json

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
    
    return render(request, 'reservation_rooms.html', {'floors': floors})

def create_reservation(request, room_id):
    room = get_object_or_404(Room, room_id=room_id)
    room.save()
    print(room.room_number)
    if request.method == 'POST':
        form = ReservationForm(request.POST)        
        if form.is_valid():
            reservation = form.save(commit=False)
            room = get_object_or_404(Room, room_id=reservation.room.room_id)
            room.is_occupied = True
            reservation.room = room
            reservation.save()

            return render(request, 'reservation_rooms.html') 
    else:
        form = ReservationForm()

    return render(request, 'create_reservation.html', {'form': form, 'room': room})

def guest_list(request):
    guests = Guest.objects.all()
    return render(request, 'management/guest/guest_list.html', {'guests': guests})

# Detail hosta
def guest_detail(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    return render(request, 'management/guest/guest_detail.html', {'guest': guest})

# Vytvoření hosta
def guest_create(request):
    return render(request, 'management/guest/guest_form.html', {'guest': request})

def add_guest(request):
    if request.method == 'POST':
        guest_form = GuestForm(request.POST)
        guest_form.guest_type = 'normal'
        address_form = AddressForm(request.POST)
        for field, errors in guest_form.errors.items():
                print(f"Chyby pro {field}: {errors}")
        for field, errors in address_form.errors.items():
            print(f"Chyby pro {field}: {errors}")

        if guest_form.is_valid() and address_form.is_valid():
            # Nejprve uložíme adresu
            address = address_form.save()
            
            # Pak přidáme hosta, přiřadíme mu právě uloženou adresu
            guest = guest_form.save(commit=False)
            guest.address = address
            guest.save()

            
            
            return redirect('guest_list')
        else:
            print("almost done")
            # Pokud formulář není platný, přidáme do kontextu chyby
            return render(request, 'forms/add_guest.html', {
                'guest_form': guest_form,
                'address_form': address_form,
                'guest_form_errors': guest_form.errors,
                'address_form_errors': address_form.errors
            })
    else:
        guest_form = GuestForm()
        address_form = AddressForm()

    return render(request, 'forms/add_guest.html', {
        'guest_form': guest_form,
        'address_form': address_form
    })

# Room ocupancies
def room_occupancies(request):
    return render(request, 'room_occupancies.html')


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
    return render(request, 'guest_form.html', {'form': form})

# Smazání hosta
def guest_delete(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    if request.method == "POST":
        guest.delete()
        return JsonResponse({'success': True})  # Vrátí JSON odpověď
    return JsonResponse({'success': False}, status=400)

# Seznam rezervací
def reservation_list(request):
    reservations = Reservation.objects.all()
    return render(request, 'reservation_list.html', {'reservations': reservations})

# Detail rezervace
def reservation_detail(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    return render(request, 'reservation_detail.html', {'reservation': reservation})

def room_management(request):
    # Získání všech místností a jejich typů
    rooms = Room.objects.select_related('room_type').all()
    return render(request, 'management/room/room_management.html', {'rooms': rooms})

def room_create(request):
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('room_management')  # Přesměrování na stránku správy pokojů
    else:
        form = RoomForm()
    return render(request, 'forms/room_form.html', {'form': form})


def room_detail(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    return render(request, 'management/room/room_detail.html', {'room': room})

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

def room_delete(request, room_id):
    room = get_object_or_404(Room, pk=room_id)
    if request.method == "POST":
        room.delete()
        return redirect('room_management')
    return render(request, 'management/room/room_confirm_delete.html', {'room': room})

# Rezerzervace 
def reservation_success(request, roomid):
    room = get_object_or_404(Room, pk=roomid)
    if room:
        room.is_occupied = True
        room.save()
        return render(request, 'reservation_success.html', {'room': room})
    
def employe_management(request):
    emp = Employee.objects.all()
    return render(request, 'management/employe_management.html', {'emp': emp})


def payment_management(request):
    payment = Payment.objects.all()
    return render(request, 'management/payment/payment_management.html', {'payment': payment})

@csrf_exempt
def mark_payment_as_paid(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            payment_id = data.get('payment_id')
            payment = Payment.objects.get(pk=payment_id)
            payment.is_paid = True
            payment.payment_date = date.today()  # Nastavení aktuálního data
            payment.save()
            return JsonResponse({'success': True})
        except Payment.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Payment not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)