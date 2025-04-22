using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace mvcapplikace.Services
{
    public class MenuService
    {
        private readonly HttpClient _httpClient;

        public MenuService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<List<MenuItem>> GetFilteredMenuAsync()
        {
            string url = "https://stravovani.vsb.cz/webkredit/Api/Ordering/Menu?Dates="
                         + DateTime.Now.Date.ToUniversalTime().ToString("O")
                         + "&CanteenId=1";
            Uri uri = new Uri(url);
            // Nastavení User-Agent hlavičky
            _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("test");

            // Volání API
            var response = await _httpClient.GetAsync(uri);
            response.EnsureSuccessStatusCode();

            // Deserializace JSON odpovědi
            var json = await response.Content.ReadAsStringAsync();
            var menu = JsonSerializer.Deserialize<MenuResponse>(json);

            // Filtrování jídel
            return menu?.Groups
                .Where(m => m.MealKindId == 2 && m.AltId > 0)
                .ToList() ?? new List<MenuItem>();
        }
    }

    // Modely pro deserializaci JSON
    public class MenuResponse
    {
        public List<MenuItem> Groups { get; set; }
    }

    public class MenuItem
    {
        [JsonPropertyName("mealKindId")]
        public int MealKindId { get; set; }
        public int AltId { get; set; }
        [JsonPropertyName("mealKindName")]
        public string Name { get; set; }
    }
}