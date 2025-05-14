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
        public UsersPage()
        {
            InitializeComponent();
            DataContext = new UsersPageViewModel();
        }

        private async void DataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is DataGrid dg && dg.SelectedItem is User selectedUser)
            {
                var dialog = new UserEditDialog(selectedUser);
                if (dialog.ShowDialog() == true)
                {
                    var userService = new UserManagement();
                    await userService.UpdateUserAsync(dialog.ToApiUser());
                    // Aktualizujte data v tabulce
                    await (DataContext as UsersPageViewModel)?.ReloadUsersAsync();
                }
            }
        }
    }
}
