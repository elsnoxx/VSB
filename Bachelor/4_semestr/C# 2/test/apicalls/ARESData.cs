using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace apicalls
{
    public static class ARESData
    {
        private static readonly HttpClient _httpClient = new HttpClient();

        public static async Task<Company> GetCompanyDataAsync(string ico)
        {
            string url = $"https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{ico}";

            try
            {
                HttpResponseMessage response = await _httpClient.GetAsync(url);

                if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    return null;
                }

                response.EnsureSuccessStatusCode();

                string jsonResponse = await response.Content.ReadAsStringAsync();

                var companyData = JsonSerializer.Deserialize<Company>(jsonResponse, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                return companyData;
            }
            catch (Exception ex)
            {
                throw new ApplicationException("Chyba při získávání dat z ARES.", ex);
            }
        }
    }
}
