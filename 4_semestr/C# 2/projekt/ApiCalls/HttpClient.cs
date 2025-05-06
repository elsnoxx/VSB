using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

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

        /// <summary>
        /// Sends a GET request to the specified endpoint.
        /// </summary>
        /// <param name="endpoint">The API endpoint (relative to the base URI).</param>
        /// <returns>The response content as a string.</returns>
        public async Task<string> GetAsync(string endpoint)
        {
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
            var content = new StringContent(data, Encoding.UTF8, "application/json");
            var response = await _httpClient.PutAsync(endpoint, content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}
