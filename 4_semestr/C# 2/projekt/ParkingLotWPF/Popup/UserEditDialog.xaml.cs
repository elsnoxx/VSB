using System.Windows;
using System.Windows.Controls;
using ApiCalls.Model;
using System.Linq;
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

        private async void DeleteCar_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is int carId)
            {
                if (MessageBox.Show("Opravdu chcete smazat toto auto?", "Potvrzení", MessageBoxButton.YesNo, MessageBoxImage.Question) == MessageBoxResult.Yes)
                {
                    var userManagement = new ApiCalls.UserManagement();
                    bool success = await userManagement.DeleteCarAsync(carId);
                    if (success)
                    {
                        MessageBox.Show("Auto bylo smazáno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                        var refreshedProfile = await userManagement.GetUserProfileAsync(Profile.id);
                        Profile = refreshedProfile;
                        DataContext = Profile;
                    }
                    else
                    {
                        MessageBox.Show("Smazání auta selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

        private async void ParkCar_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is CarDto car)
            {
                // 1. Vyber parkoviště (např. zobraz dialog s výběrem)
                var parkingLotSelection = new SelectParkingLotDialog();
                if (parkingLotSelection.ShowDialog() == true)
                {
                    int selectedParkingLotId = parkingLotSelection.SelectedParkingLotId;

                    // 2. Zavolej API pro obsazení místa
                    var manager = new ApiCalls.ParkingSpaceManagement();
                    var occupyRequest = new OccupyRequest { licensePlate = car.licensePlate };
                    var result = await manager.OccupySpaceAsync(selectedParkingLotId, occupyRequest);

                    if (result != null)
                    {
                        MessageBox.Show($"Auto zaparkováno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);
                        
                        var userManagement = new ApiCalls.UserManagement();
                        var refreshedProfile = await userManagement.GetUserProfileAsync(Profile.id);
                        Profile = refreshedProfile;
                        DataContext = Profile;
                    }
                    else
                    {
                        MessageBox.Show("Zaparkování selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }

        private async void UnparkCar_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.Tag is CurrentParkingDto currentParking)
            {
                var manager = new ApiCalls.ParkingSpaceManagement();
                var releaseRequest = new ReleaseRequest
                {
                    ParkingSpaceId = currentParking.parkingSpaceId,
                    ParkingLotId = currentParking.parkingLotId
                };
                bool success = await manager.ReleaseSpaceAsync(releaseRequest);

                if (success)
                {
                    MessageBox.Show("Auto bylo odparkováno.", "Hotovo", MessageBoxButton.OK, MessageBoxImage.Information);

                    var userManagement = new ApiCalls.UserManagement();
                    var refreshedProfile = await userManagement.GetUserProfileAsync(Profile.id);
                    Profile = refreshedProfile;
                    DataContext = Profile;
                }
                else
                {
                    MessageBox.Show("Odparkování selhalo.", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                }
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