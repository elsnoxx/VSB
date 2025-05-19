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
using System.Windows.Navigation;
using System.Windows.Shapes;
using ApiCalls.Model;
using ParkingLotWPF.Popup;


namespace ParkingLotWPF.Views
{
    /// <summary>
    /// Interaction logic for ParkingLotsPage.xaml
    /// </summary>
    public partial class ParkingLotsPage : UserControl
    {
        private readonly ParkinglotPageViewModel _viewModel;

        public ParkingLotsPage()
        {
            InitializeComponent();
            _viewModel = new ParkinglotPageViewModel();
            DataContext = _viewModel;
        }

        private async void DataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is DataGrid dg && dg.SelectedItem is User selectedUser)
            {
                var userManagement = new ApiCalls.ParkinglotManagement();
                var profile = await userManagement.GetAllParkinglot(selectedUser.Id);

                // Otevření dialogu s kompletním profilem
                var dialog = new ParkinglotEditDialog(profile);
                if (dialog.ShowDialog() == true)
                {
                    // případné uložení změn
                }
            }
        }
    }

}
