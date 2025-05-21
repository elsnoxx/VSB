using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Configuration; // přidej using

namespace ApiCalls
{
    public class HttpClientWrapper
    {
        private readonly HttpClient _httpClient;

        public HttpClientWrapper()
        {
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri("http://localhost:5062/")
            };
        }

        private void AddApiKeyHeader()
        {
            var apiKey = ConfigurationManager.AppSettings["ApiKey"];
            Console.WriteLine($"[DEBUG] ApiKey z configu: '{apiKey}'");
            if (_httpClient.DefaultRequestHeaders.Contains("X-Api-Key"))
                _httpClient.DefaultRequestHeaders.Remove("X-Api-Key");
            _httpClient.DefaultRequestHeaders.Add("X-Api-Key", apiKey);
        }

        /// <summary>
        /// Sends a GET request to the specified endpoint.
        /// </summary>
        /// <param name="endpoint">The API endpoint (relative to the base URI).</param>
        /// <returns>The response content as a string.</returns>
        public async Task<string> GetAsync(string endpoint)
        {
            AddApiKeyHeader();
            var response = await _httpClient.GetAsync(endpoint);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }

        /// <summary>
        /// Sends a POST request to the specified endpoint with the provided data.
        /// </summary>
        /// <param name="endpoint">The API endpoint (relative to the base URI).</param>
        /// <param name="data">The data to send in the request body.</param>
        /// <returns>The response content as a string.</returns>
        public async Task<string> PostAsync(string endpoint, string data)
        {
            AddApiKeyHeader();
            var content = new StringContent(data, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(endpoint, content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }

        /// <summary>
        /// Sends a DELETE request to the specified endpoint.
        /// </summary>
        /// <param name="endpoint">The API endpoint (relative to the base URI).</param>
        /// <returns>The response content as a string.</returns>
        public async Task<string> DeleteAsync(string endpoint)
        {
            AddApiKeyHeader();
            var response = await _httpClient.DeleteAsync(endpoint);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }

        /// <summary>
        /// Sends a PUT request to the specified endpoint with the provided data.
        /// </summary>
        /// <param name="endpoint">The API endpoint (relative to the base URI).</param>
        /// <param name="data">The data to send in the request body.</param>
        /// <returns>The response content as a string.</returns>
        public async Task<string> PutAsync(string endpoint, string data)
        {
            AddApiKeyHeader();
            var content = new StringContent(data, Encoding.UTF8, "application/json");
            var response = await _httpClient.PutAsync(endpoint, content);
            var responseBody = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
            {
                // Vypište detail chyby do konzole
                Console.WriteLine($"API ERROR: {response.StatusCode} - {responseBody}");
                throw new HttpRequestException($"API ERROR: {response.StatusCode} - {responseBody}");
            }
            return responseBody;
        }
    }
}
