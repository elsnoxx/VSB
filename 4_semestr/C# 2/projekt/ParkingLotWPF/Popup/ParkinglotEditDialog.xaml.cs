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

        public async void EditParkingLot_Click(object sender, RoutedEventArgs e)
        {
            var parkinglot = ToApParkinglot();
            var editDialog = new ParkingEditDialog(parkinglot);
            editDialog.ShowDialog();
        }

        private async void EditParkingSpace_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is ParkingSpace selectedSpace)
            {
                var dialog = new ParkingSpaceEditDialog(selectedSpace);
                if (dialog.ShowDialog() == true)
                {
                    var manager = new ApiCalls.ParkingSpaceManagement();
                    bool success = await manager.UpdateSpaceAsync(selectedSpace);
                    if (success)
                        MessageBox.Show("Změny byly uloženy.", "Hotovo");
                    else
                        MessageBox.Show("Uložení selhalo.", "Chyba");
                }
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
    }
}
