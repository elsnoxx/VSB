using ApiCalls.Model;
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
    /// Interaction logic for UserDataEditing.xaml
    /// </summary>
    public partial class UserDataEditing : Window
    {
        public List<string> Roles { get; } = new List<string> { "User", "Admin" };
        private User _user;

        public UserDataEditing(User user)
        {
            InitializeComponent();
            _user = user;
            DataContext = _user;
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            this.DialogResult = false;
            this.Close();
        }

        private bool ValidateUser(User user)
        {
            if (string.IsNullOrWhiteSpace(user.FirstName) || user.FirstName.Length > 50)
            {
                MessageBox.Show("Jméno je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (string.IsNullOrWhiteSpace(user.LastName) || user.LastName.Length > 50)
            {
                MessageBox.Show("Příjmení je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (string.IsNullOrWhiteSpace(user.Username) || user.Username.Length > 50)
            {
                MessageBox.Show("Uživatelské jméno je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (string.IsNullOrWhiteSpace(user.Email) || user.Email.Length > 100)
            {
                MessageBox.Show("Email je povinný a může mít maximálně 100 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            // Validace emailu
            try
            {
                var addr = new System.Net.Mail.MailAddress(user.Email);
                if (addr.Address != user.Email)
                    throw new Exception();
            }
            catch
            {
                MessageBox.Show("Zadejte platný email.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            if (string.IsNullOrWhiteSpace(user.Role))
            {
                MessageBox.Show("Role je povinná.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }
            return true;
        }


        private async void Update_Click(object sender, RoutedEventArgs e)
        {
            if (!ValidateUser(_user))
                return;

            var userManagement = new ApiCalls.UserManagement();
            try
            {
                await userManagement.UpdateUserAsync(_user);
                MessageBox.Show("Uživatel byl úspěšně aktualizován.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                DialogResult = true;
                Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Chyba při aktualizaci uživatele: " + ex.Message, "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

    }
}
