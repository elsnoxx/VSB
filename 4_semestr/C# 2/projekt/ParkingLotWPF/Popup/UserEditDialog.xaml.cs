using System.Windows;
using ApiCalls.Model;

namespace ParkingLotWPF.Views
{
    public partial class UserEditDialog : Window
    {
        private readonly User _originalUser;

        public User EditUser { get; private set; }

        public UserEditDialog(User user)
        {
            InitializeComponent();
            _originalUser = user;
            EditUser = new User
            {
                Id = user.Id,
                Username = user.Username,
                FirstName = user.FirstName,
                LastName = user.LastName,
                Email = user.Email
            };
            DataContext = EditUser;
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        public User ToApiUser()
        {
            return new User
            {
                Id = EditUser.Id,
                Username = EditUser.Username,
                FirstName = EditUser.FirstName,
                LastName = EditUser.LastName,
                Email = EditUser.Email,
                Role = _originalUser.Role,
                Password = _originalUser.Password
            };
        }
    }
}