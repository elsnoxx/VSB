using ApiCalls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using ApiCalls.Model;

namespace ParkingLotWPF.Popup
{
    /// <summary>
    /// Interaction logic for CarRegistrationDialog.xaml
    /// </summary>
    public partial class CarRegistrationDialog : Window
    {
        private readonly int _userId;

        public CarRegistrationDialog(int userId)
        {
            InitializeComponent();
            _userId = userId;
        }

        private async void Register_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(LicensePlateBox.Text) ||
                string.IsNullOrWhiteSpace(BrandModelBox.Text) ||
                string.IsNullOrWhiteSpace(ColorBox.Text))
            {
                MessageBox.Show("Všechna pole musí být vyplněna.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var carDto = new CarDto
            {
                userId = _userId,
                licensePlate = LicensePlateBox.Text,
                brandModel = BrandModelBox.Text,
                color = ColorBox.Text
            };

            var userManagement = new UserManagement();
            bool success = await userManagement.CreateCarAsync(carDto);
            if (success)
            {
                MessageBox.Show("Auto bylo úspěšně zaregistrováno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Registrace auta selhala.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
