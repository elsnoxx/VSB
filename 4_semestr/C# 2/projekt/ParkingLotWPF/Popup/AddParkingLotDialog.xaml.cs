using System.Windows;
using ApiCalls.Model;
using ApiCalls;

namespace ParkingLotWPF.Popup
{
    public partial class AddParkingLotDialog : Window
    {
        public Parkinglot NewParkingLot { get; private set; }

        public AddParkingLotDialog()
        {
            InitializeComponent();
        }

        private async void Create_Click(object sender, RoutedEventArgs e)
        {
            // Validace vstupů
            if (string.IsNullOrWhiteSpace(NameBox.Text) ||
                string.IsNullOrWhiteSpace(LatitudeBox.Text) ||
                string.IsNullOrWhiteSpace(LongitudeBox.Text) ||
                string.IsNullOrWhiteSpace(CapacityBox.Text))
            {
                MessageBox.Show("Všechna pole musí být vyplněna.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!decimal.TryParse(LatitudeBox.Text, out decimal latitude) ||
                !decimal.TryParse(LongitudeBox.Text, out decimal longitude) ||
                !int.TryParse(CapacityBox.Text, out int capacity))
            {
                MessageBox.Show("Zadejte platné číselné hodnoty pro Latitude, Longitude a Kapacitu.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var lot = new Parkinglot
            {
                name = NameBox.Text,
                latitude = latitude,
                longitude = longitude,
                capacity = capacity,
                freeSpaces = capacity
            };

            NewParkingLot = lot;
            DialogResult = true;
            Close();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
