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
    /// Interaction logic for UserRegistrationDialog.xaml
    /// </summary>
    public partial class UserRegistrationDialog : Window
    {
        public User NewUser { get; private set; }

        public UserRegistrationDialog()
        {
            InitializeComponent();
        }

        private void Register_Click(object sender, RoutedEventArgs e)
        {
            var selectedRole = (RoleComboBox.SelectedItem as ComboBoxItem)?.Content as string;

            // Validace
            if (string.IsNullOrWhiteSpace(FirstNameBox.Text) || FirstNameBox.Text.Length > 50)
            {
                MessageBox.Show("Jméno je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrWhiteSpace(LastNameBox.Text) || LastNameBox.Text.Length > 50)
            {
                MessageBox.Show("Příjmení je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrWhiteSpace(UsernameBox.Text) || UsernameBox.Text.Length > 50)
            {
                MessageBox.Show("Uživatelské jméno je povinné a může mít maximálně 50 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrWhiteSpace(PasswordBox.Password) || PasswordBox.Password.Length > 100)
            {
                MessageBox.Show("Heslo je povinné a může mít maximálně 100 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrWhiteSpace(selectedRole))
            {
                MessageBox.Show("Role je povinná.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (string.IsNullOrWhiteSpace(EmailBox.Text) || EmailBox.Text.Length > 100)
            {
                MessageBox.Show("Email je povinný a může mít maximálně 100 znaků.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            // Validace emailu
            try
            {
                var addr = new System.Net.Mail.MailAddress(EmailBox.Text);
                if (addr.Address != EmailBox.Text)
                    throw new Exception();
            }
            catch
            {
                MessageBox.Show("Zadejte platný email.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            NewUser = new User
            {
                FirstName = FirstNameBox.Text,
                LastName = LastNameBox.Text,
                Username = UsernameBox.Text,
                Email = EmailBox.Text,
                Password = PasswordBox.Password,
                Role = selectedRole
            };
            DialogResult = true;
            Close();
        }


        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
