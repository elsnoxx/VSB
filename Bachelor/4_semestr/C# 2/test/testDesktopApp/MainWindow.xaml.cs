using System.Collections.ObjectModel;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using apicalls;

namespace testDesktopApp;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public ObservableCollection<CompanyItem> CompanyList { get; set; }
    public MainWindow()
    {
        InitializeComponent();
        CompanyList = new ObservableCollection<CompanyItem>();
        DataContext = this;
    }

    private void textChangeInputIco(object sender, TextChangedEventArgs e)
    {
        if (inputIco.Text.Length == 0)
        {
            AddIco.IsEnabled = false;
        }
        else
        {
            AddIco.IsEnabled = true;
        }
    }

    private async void AddNewConpany(object sender, RoutedEventArgs e)
    {
        try
        {
            string ico = inputIco.Text.Trim();
            var company = await ARESData.GetCompanyDataAsync(ico);

            if (company == null)
            {
                MessageBox.Show($"Firma s IČO {inputIco.Text} nebyla nalezena.", "Info", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            else
            {
                editWindow editWindow = new editWindow(company, CompanyList);
                editWindow.DataContext = company;
                editWindow.Show();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Došlo k chybě: {ex.Message}");
        }
    }
}