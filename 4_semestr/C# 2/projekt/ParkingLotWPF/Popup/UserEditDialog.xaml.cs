using System.Windows;
using ApiCalls.Model;
using ParkingLotWPF.Popup;

namespace ParkingLotWPF.Views
{
    public partial class UserEditDialog : Window
    {
        public UserProfileViewModel Profile { get; private set; }

        public UserEditDialog(UserProfileViewModel profile)
        {
            InitializeComponent();
            Profile = profile;
            DataContext = Profile;
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        public async void Reset_Password(object sender, RoutedEventArgs e)
        {
            var dialog = new PasswordResetDialog();
            if (dialog.ShowDialog() == true)
            {
                string newPassword = dialog.NewPassword;
                var userManagement = new ApiCalls.UserManagement();
                bool success = await userManagement.ResetPasswordAsync(Profile.id, newPassword);
                if (success)
                    MessageBox.Show("Heslo bylo úspěšně změněno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                else
                    MessageBox.Show("Chyba při změně hesla.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Edit_User(object sender, RoutedEventArgs e)
        {
            var user = ToApiUser();
            var editDialog = new UserDataEditing(user);
            editDialog.ShowDialog();
        }

        private async void Add_Car(object sender, RoutedEventArgs e)
        {
            var user = ToApiUser();
            var editDialog = new CarRegistrationDialog(user.Id);
            if (editDialog.ShowDialog() == true)
            {
                var userManagement = new ApiCalls.UserManagement();
                var refreshedProfile = await userManagement.GetUserProfileAsync(user.Id);
                Profile = refreshedProfile;
                DataContext = Profile;
            }
        }


        public User ToApiUser()
        {
            return new User
            {
                Id = Profile.id,
                Username = Profile.username,
                FirstName = Profile.firstName,
                LastName = Profile.lastName,
                Email = Profile.email,
                Role = Profile.role,
                Password = Profile.password
            };
        }

    }
}