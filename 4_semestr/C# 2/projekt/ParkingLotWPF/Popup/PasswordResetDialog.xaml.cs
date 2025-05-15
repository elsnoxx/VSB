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
    /// Interaction logic for PasswordResetDialog.xaml
    /// </summary>
    public partial class PasswordResetDialog : Window
    {
        public string NewPassword { get; private set; }
        public PasswordResetDialog()
        {
            InitializeComponent();
        }
        private bool ValidatePassword(string password, out string error)
        {
            error = null;
            if (password.Length < 8)
            {
                error = "Heslo musí mít alespoň 8 znaků.";
                return false;
            }
            if (!password.Any(char.IsDigit))
            {
                error = "Heslo musí obsahovat alespoň jedno číslo.";
                return false;
            }
            if (!password.Any(char.IsUpper))
            {
                error = "Heslo musí obsahovat alespoň jedno velké písmeno.";
                return false;
            }
            if (!password.Any(ch => !char.IsLetterOrDigit(ch)))
            {
                error = "Heslo musí obsahovat alespoň jeden speciální znak.";
                return false;
            }
            return true;
        }


        private void Ok_Click(object sender, RoutedEventArgs e)
        {
            var newPassword = PasswordBoxNew.Password;
            var confirmPassword = PasswordBoxConfirm.Password;
            if (string.IsNullOrWhiteSpace(newPassword))
            {
                MessageBox.Show("Zadejte nové heslo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (newPassword != confirmPassword)
            {
                MessageBox.Show("Hesla se neshodují.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            if (!ValidatePassword(newPassword, out string error))
            {
                MessageBox.Show(error, "Chyba", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
            NewPassword = newPassword;
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
