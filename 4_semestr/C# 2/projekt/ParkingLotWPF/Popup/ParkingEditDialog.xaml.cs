using ApiCalls;
using ApiCalls.Model;
using System;
using System.Threading.Tasks;
using System.Windows;

namespace ParkingLotWPF.Popup
{
    public partial class ParkingEditDialog : Window
    {
        private readonly ParkinglotManagement _management = new ParkinglotManagement();
        private readonly Parkinglot _parkinglot;

        public ParkingEditDialog(Parkinglot parkinglot)
        {
            InitializeComponent();
            _parkinglot = parkinglot;
            NameBox.Text = parkinglot.name;
            LatitudeBox.Text = parkinglot.latitude.ToString();
            LongitudeBox.Text = parkinglot.longitude.ToString();
        }

        public async void Save_Click(object sender, RoutedEventArgs e)
        {
            ErrorText.Text = "";

            // Validace
            if (string.IsNullOrWhiteSpace(NameBox.Text))
            {
                ErrorText.Text = "Název je povinný.";
                return;
            }
            if (!decimal.TryParse(LatitudeBox.Text, out decimal latitude))
            {
                ErrorText.Text = "Latitude musí být číslo.";
                return;
            }
            if (!decimal.TryParse(LongitudeBox.Text, out decimal longitude))
            {
                ErrorText.Text = "Longitude musí být číslo.";
                return;
            }

            // Aktualizace objektu
            _parkinglot.name = NameBox.Text;
            _parkinglot.latitude = latitude;
            _parkinglot.longitude = longitude;

            // API volání
            bool success = await _management.UpdateParkinglotAsync(_parkinglot);
            if (success)
            {
                DialogResult = true;
                Close();
            }
            else
            {
                ErrorText.Text = "Uložení selhalo.";
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}