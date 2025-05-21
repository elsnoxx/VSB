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
using ApiCalls;
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
            if (sender is DataGrid dg && dg.SelectedItem is Parkinglot selectedParkinglot)
            {
                var userManagement = new ApiCalls.ParkinglotManagement();
                var profileParkinglot = await userManagement.GetParkinglotByIdAsync(selectedParkinglot.parkingLotId);

                // Otevření dialogu s kompletním profilem
                var dialog = new ParkinglotEditDialog(profileParkinglot);
                if (dialog.ShowDialog() == true)
                {
                    await _viewModel.LoadParkinglotsAsync();
                }
            }
        }

        private async void AddParkingLot_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new AddParkingLotDialog();
            dialog.ShowDialog();
            if (dialog.DialogResult == true)
            {
                var newParkinglot = dialog.NewParkingLot;
                var parkinglotManagement = new ApiCalls.ParkinglotManagement();
                bool success = await parkinglotManagement.CreateParkinglotAsync(newParkinglot);
                if (success)
                {
                    MessageBox.Show("Parkoviště bylo úspěšně vytvořeno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                    await _viewModel.LoadParkinglotsAsync();
                }
                else
                {
                    MessageBox.Show("Vytvoření parkoviště selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }

        }

        private async void DeleteCar_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is int parkingLotId)
            {
                if (MessageBox.Show("Opravdu chcete smazat toto parkoviště?", "Potvrzení", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
                {
                    var service = new ParkinglotManagement();
                    bool success = await service.DeleteParkinglotAsync(parkingLotId);
                    if (success)
                    {
                        MessageBox.Show("Parkoviště bylo smazáno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                        await _viewModel.LoadParkinglotsAsync();
                    }
                    else
                    {
                        MessageBox.Show("Smazání parkoviště selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

        private async void ChangePrice_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.DataContext is Parkinglot lot)
            {
                var dialog = new ChangePricePerHourDialog(lot.pricePerHour);
                if (dialog.ShowDialog() == true)
                {
                    var newPrice = dialog.Price;
                    var service = new ApiCalls.ParkinglotManagement();
                    bool success = await service.UpdatePricePerHourAsync(lot.parkingLotId, newPrice);
                    if (success)
                    {
                        MessageBox.Show("Cena byla změněna.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                        await _viewModel.LoadParkinglotsAsync();
                    }
                    else
                    {
                        MessageBox.Show("Změna ceny selhala.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

    }

}
