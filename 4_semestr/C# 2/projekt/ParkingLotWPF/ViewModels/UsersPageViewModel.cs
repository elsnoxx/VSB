using System.Collections.ObjectModel;
using ApiCalls.Model;
using ApiCalls;
using System.Threading.Tasks;
using System.ComponentModel;

public class UsersPageViewModel : INotifyPropertyChanged
{
    public ObservableCollection<User> Users { get; set; } = new ObservableCollection<User>();

    public UsersPageViewModel()
    {
        LoadUsersAsync();
    }

    private async void LoadUsersAsync()
    {
        await ReloadUsersAsync();
    }

    public async Task ReloadUsersAsync()
    {
        var userService = new UserManagement();
        var users = await userService.GetAllUsersAsync();
        Users = new ObservableCollection<User>(users);
        OnPropertyChanged(nameof(Users));
    }

    public event PropertyChangedEventHandler PropertyChanged;
    protected void OnPropertyChanged(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}