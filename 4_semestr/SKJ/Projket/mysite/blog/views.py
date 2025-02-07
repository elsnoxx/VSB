from django.shortcuts import render, get_object_or_404, redirect
from .models import Guest, Reservation
from .forms import GuestForm, ReservationForm

# Hlavní stránka
def index(request):
    return render(request, 'base.html')

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
    if request.method == "POST":
        form = GuestForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('guest_list')  # Po úspěšném uložení přesměrování
    else:
        form = GuestForm()

    return render(request, 'forms/add_guest.html', {'form': form})

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
