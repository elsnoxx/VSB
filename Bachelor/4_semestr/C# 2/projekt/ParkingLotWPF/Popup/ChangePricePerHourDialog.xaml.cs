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

namespace ParkingLotWPF.Popup
{
    /// <summary>
    /// Interaction logic for ChangePricePerHourDialog.xaml
    /// </summary>
    public partial class ChangePricePerHourDialog : Window
    {
        public decimal Price { get; private set; }
        public ChangePricePerHourDialog(decimal currentPrice)
        {
            InitializeComponent();
            PriceBox.Text = currentPrice.ToString("0.##");
            PriceBox.Focus();
        }

        private void Ok_Click(object sender, RoutedEventArgs e)
        {
            if (decimal.TryParse(PriceBox.Text, out var price) && price >= 0)
            {
                Price = price;
                DialogResult = true;
                Close();
            }
            else
            {
                MessageBox.Show("Zadejte platnou cenu.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
