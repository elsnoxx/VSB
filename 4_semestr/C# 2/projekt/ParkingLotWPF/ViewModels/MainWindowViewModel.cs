using System.ComponentModel;
using System.Windows.Input;
using ParkingLotWPF.Views;

public class MainWindowViewModel : INotifyPropertyChanged
{
    public ICommand ShowUsersCommand { get; }
    public ICommand ShowParkingLotsCommand { get; }

    private object _currentPage;
    public object CurrentPage
    {
        get => _currentPage;
        set { _currentPage = value; OnPropertyChanged(nameof(CurrentPage)); }
    }

    public MainWindowViewModel()
    {
        ShowUsersCommand = new RelayCommand(_ => ShowUsers());
        ShowParkingLotsCommand = new RelayCommand(_ => ShowParkingLots());
        ShowUsers();
    }

    private void ShowUsers()
    {
        CurrentPage = new UsersPage();
    }

    private void ShowParkingLots()
    {
        CurrentPage = new ParkingLotsPage();
    }

    public event PropertyChangedEventHandler PropertyChanged;
    protected void OnPropertyChanged(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}