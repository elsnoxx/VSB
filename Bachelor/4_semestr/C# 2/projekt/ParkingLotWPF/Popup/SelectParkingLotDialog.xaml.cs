using System.Collections.Generic;
using System.Windows;
using ApiCalls.Model;

namespace ParkingLotWPF.Popup
{
    public partial class SelectParkingLotDialog : Window
    {
        public int SelectedParkingLotId { get; private set; }

        public SelectParkingLotDialog()
        {
            InitializeComponent();
            LoadParkingLots();
        }

        private async void LoadParkingLots()
        {
            var manager = new ApiCalls.ParkinglotManagement();
            var lots = await manager.GetAllParkinglotsAsync();
            ParkingLotCombo.ItemsSource = lots;
        }

        private void Ok_Click(object sender, RoutedEventArgs e)
        {
            if (ParkingLotCombo.SelectedItem is Parkinglot selected)
            {
                SelectedParkingLotId = selected.parkingLotId;
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Vyberte parkoviště.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}