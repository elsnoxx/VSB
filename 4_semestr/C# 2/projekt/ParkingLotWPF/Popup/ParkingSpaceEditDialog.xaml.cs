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
            StatusCombo.ItemsSource = new List<string> { "available", "maintenance", "occupied" };
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_space.status))
            {
                MessageBox.Show("Musíte vybrat stav parkovacího místa.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            DialogResult = true;
            Close();
        }
    }

    
}
