using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Text.Json;
using System.Windows.Markup;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }
        private void TextBox1_TextChanged(object sender, TextChangedEventArgs e)
        {
            // Handle text changed event
            if (TextBox1.Text.Length > 0)
            {
                GetIco.IsEnabled = true;
            }
            else
            {
                GetIco.IsEnabled = false;
            }
        }

        private async void GetIco_Click(object sender, RoutedEventArgs e)
        {

            
            string ico = TextBox1.Text;
            JsonData info = await getInfo(ico);
            Window1 window1 = new Window1();
            window1.DataContext = info;
            window1.Show();

        }

        public async Task<JsonData> getInfo(string ico)
        {
            try
            {
                Uri url = new Uri("https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/" + ico);
                HttpClient client = new HttpClient();
                var response = client.GetAsync(url);

                if (response.Result.IsSuccessStatusCode)
                {
                    var options = new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    };

                    string jsonResponse = await response.Result.Content.ReadAsStringAsync();

                    JsonData jsonData = JsonSerializer.Deserialize<JsonData>(jsonResponse, options);
                    return jsonData;
                }
                else
                {
                    MessageBox.Show("Error: " + response.Result.StatusCode.ToString(), "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    return null;
                }
            }
            catch (Exception)
            {

                throw;
            }
        }
    }
}
