using ApiCalls.Model;
using System.Windows;
using System.Windows.Controls;

namespace ParkingLotWPF.Popup
{
    public partial class ParkinglotEditDialog : Window
    {
        private readonly ParkingLotProfileViewModel _profile;

        public ParkinglotEditDialog(ParkingLotProfileViewModel profile)
        {
            InitializeComponent();
            _profile = profile;
            DataContext = _profile;
        }

        private void EditParkingSpace_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is ParkingSpace space)
            {
                var dialog = new ParkingSpaceEditDialog(space);
                if (dialog.ShowDialog() == true)
                {
                    // Refresh dat, pokud je potřeba
                }
            }
        }

        public async void EditParkingLot_Click(object sender, RoutedEventArgs e)
        {
            var parkinglot = ToApParkinglot();
            var editDialog = new ParkingLotDataEditing(parkinglot);
            if (editDialog.ShowDialog() == true)
            {
                // případně refresh dat
            }
        }

        public Parkinglot ToApParkinglot()
        {
            return new Parkinglot
            {
                parkingLotId = _profile.parkingLotId,
                name = _profile.name,
                latitude = _profile.latitude,
                longitude = _profile.longitude,
                capacity = _profile.capacity,
                freeSpaces = _profile.freeSpaces
            };
        }

        public ParkingSpace ToApiParkingSpace()
        {
            return new ParkingSpace
            {
                parkingSpaceId = _profile.parkingSpaces[0].parkingSpaceId,
                status = _profile.parkingSpaces[0].status,
                parkingLotId = _profile.parkingLotId
            };
        }
    }
}
