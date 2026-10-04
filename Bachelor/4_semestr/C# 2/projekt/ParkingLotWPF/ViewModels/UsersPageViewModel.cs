using System.Collections.ObjectModel;
using ApiCalls.Model;
using ApiCalls;
using System.Threading.Tasks;
using System.ComponentModel;

public class ParkinglotPageViewModel : INotifyPropertyChanged
{
    public ObservableCollection<Parkinglot> ParkingLots { get; set; } = new ObservableCollection<Parkinglot>();

    public ParkinglotPageViewModel()
    {
        LoadParkinglotsAsync();
    }

    public async Task LoadParkinglotsAsync()
    {
        await ReloadParkinglotsAsync();
    }

    public async Task ReloadParkinglotsAsync()
    {
        var parkinglotService = new ParkinglotManagement();
        var lots = await parkinglotService.GetAllParkinglotsAsync();
        ParkingLots = new ObservableCollection<Parkinglot>(lots);
        OnPropertyChanged(nameof(ParkingLots));
    }

    public event PropertyChangedEventHandler PropertyChanged;
    protected void OnPropertyChanged(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
