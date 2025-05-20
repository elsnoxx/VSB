using ApiCalls.Model;
using System;
using System.Windows;

namespace ParkingLotWPF.Popup
{
    public partial class ParkingLotDataEditing : Window
    {
        private Parkinglot _parkinglot;

        public ParkingLotDataEditing(Parkinglot parkinglot)
        {
            InitializeComponent();
            _parkinglot = parkinglot;
            DataContext = _parkinglot;
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            this.DialogResult = false;
            this.Close();
        }

        private bool ValidateParkinglot(Parkinglot lot)
        {
            if (string.IsNullOrWhiteSpace(lot.name))
            {
                MessageBox.Show("Název je povinný.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (lot.latitude < -90 || lot.latitude > 90)
            {
                MessageBox.Show("Latitude musí být mezi -90 a 90.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (lot.longitude < -180 || lot.longitude > 180)
            {
                MessageBox.Show("Longitude musí být mezi -180 a 180.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (lot.capacity < 0)
            {
                MessageBox.Show("Kapacita musí být nezáporné číslo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            return true;
        }

        private async void Update_Click(object sender, RoutedEventArgs e)
        {
            if (!ValidateParkinglot(_parkinglot))
                return;

            var manager = new ApiCalls.ParkinglotManagement();
            try
            {
                await manager.UpdateParkinglotAsync(_parkinglot);
                MessageBox.Show("Parkoviště bylo úspěšně aktualizováno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                this.DialogResult = true;
                this.Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Chyba při aktualizaci parkoviště: " + ex.Message, "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}