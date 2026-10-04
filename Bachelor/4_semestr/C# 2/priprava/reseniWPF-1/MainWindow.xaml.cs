using System;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using System.Text.Json;
using System.Collections.Generic;
using System.Windows.Controls.Primitives;


namespace reseniWPF_1
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

        private async void SubmitButton_Click(object sender, RoutedEventArgs e)
        {
            InputBox.Text = "61989100";
            InputBox.Text = InputBox.Text.Trim();
            if (string.IsNullOrEmpty(InputBox.Text))
            {
                ErrorText.Text = "Toto pole je povinne";
                ErrorText.Visibility = Visibility.Visible;
            }
            else
            {
                if (ErrorText.Visibility == Visibility.Visible)
                {
                    ErrorText.Visibility = Visibility.Hidden;
                }

                // Call the API with the input value
                var result = await getInfoFromAPI(InputBox.Text);

                Window1 window1 = new Window1(result);
                window1.Owner = this;
                window1.WindowStartupLocation = WindowStartupLocation.CenterOwner;
                window1.ShowDialog();


                //MessageBox.Show( result.ToString() );
            }

        }

        public async Task<JsonData> getInfoFromAPI(string ico)
        {
            try
            {
                Uri url = new Uri("https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/" + ico);
                using (var httpClient = new HttpClient())
                {
                    var response = await httpClient.GetAsync(url);

                    if (response.StatusCode == HttpStatusCode.NotFound)
                    {
                        ErrorText.Text = "Zadané IČO neexistuje";
                        ErrorText.Visibility = Visibility.Visible;
                        return null;
                    }

                    var options = new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    };

                    string jsonResponse = await response.Content.ReadAsStringAsync();

                    JsonData jsonData = JsonSerializer.Deserialize<JsonData>(jsonResponse, options);
                    return jsonData;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Výjimka: {ex.Message}", "Chyba", MessageBoxButton.OK, MessageBoxImage.Error);
                return null;
            }
        }
    }
}
