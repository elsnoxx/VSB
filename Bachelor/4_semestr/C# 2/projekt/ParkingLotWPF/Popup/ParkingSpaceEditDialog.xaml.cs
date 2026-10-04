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
    /// Interaction logic for ParkingSpaceEditDialog.xaml
    /// </summary>
    public partial class ParkingSpaceEditDialog : Window
    {
        private readonly ParkingSpace _space;

        public ParkingSpaceEditDialog(ParkingSpace space)
        {
            InitializeComponent();
            _space = space;
            DataContext = _space;
            StatusCombo.ItemsSource = new List<string> { "available", "under_maintenance", "occupied" };

            if (_space.status == "occupied")
            {
                StatusCombo.IsEnabled = false;
            }
        }

        private async void Save_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_space.status))
            {
                MessageBox.Show("Musíte vybrat stav parkovacího místa.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var manager = new ApiCalls.ParkingSpaceManagement();
            bool success = await manager.UpdateSpaceStatusAsync(_space.parkingSpaceId, _space.status);

            if (success)
            {
                MessageBox.Show("Stav byl úspěšně uložen.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Uložení selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

    }


}
