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
using ApiCalls;

namespace ParkingLotWPF.Views
{
    /// <summary>
    /// Interaction logic for UsersPage.xaml
    /// </summary>
    public partial class UsersPage : UserControl
    {
        private readonly UsersPageViewModel _viewModel;
        public UsersPage()
        {
            InitializeComponent();
            DataContext = new UsersPageViewModel();
        }

        private async void AddUser_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new ParkingLotWPF.Popup.UserRegistrationDialog();
            if (dialog.ShowDialog() == true)
            {
                var user = dialog.NewUser;
                var userManagement = new ApiCalls.UserManagement();
                bool success = await userManagement.CreateUserAsync(user);
                if (success)
                {
                    MessageBox.Show("Uživatel byl úspěšně zaregistrován.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                    await _viewModel.LoadUsersAsync();
                }
                else
                {
                    MessageBox.Show("Registrace selhala.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async void DataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is DataGrid dg && dg.SelectedItem is User selectedUser)
            {
                var userManagement = new ApiCalls.UserManagement();
                var profile = await userManagement.GetUserProfileAsync(selectedUser.Id);

                // Otevření dialogu s kompletním profilem
                var dialog = new UserEditDialog(profile);
                if (dialog.ShowDialog() == true)
                {
                    // případné uložení změn
                }
            }
        }

    }
}
