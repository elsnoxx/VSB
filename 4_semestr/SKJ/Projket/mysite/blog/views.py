from django.shortcuts import render, get_object_or_404, redirect
from .models import Guest, Reservation, Room
from .forms import GuestForm, ReservationForm, AddressForm

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
    print(room.room_number)
    if request.method == 'POST':
        form = ReservationForm(request.POST)
        if form.is_valid():
            reservation = form.save(commit=False)
            reservation.room = room
            reservation.save()
            return redirect('reservation_success')  # Přesměrování na stránku potvrzení
    else:
        form = ReservationForm()

    return render(request, 'create_reservation.html', {'form': form, 'room': room})

def guest_list(request):
    guests = Guest.objects.all()
    return render(request, 'guest_list.html', {'guests': guests})

# Detail hosta
def guest_detail(request, guest_id):
    guest = get_object_or_404(Guest, pk=guest_id)
    return render(request, 'guest_detail.html', {'guest': guest})

# Vytvoření hosta
def guest_create(request):
    return render(request, 'guest_form.html', {'guest': request})

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
            guest.address = address  # přiřadíme adresu
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
        return redirect('guest_list')
    return render(request, 'guest_confirm_delete.html', {'guest': guest})

# Seznam rezervací
def reservation_list(request):
    reservations = Reservation.objects.all()
    return render(request, 'reservation_list.html', {'reservations': reservations})

# Detail rezervace
def reservation_detail(request, reservation_id):
    reservation = get_object_or_404(Reservation, pk=reservation_id)
    return render(request, 'reservation_detail.html', {'reservation': reservation})
