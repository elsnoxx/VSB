using System;
using System.Text.Json;
using System.Threading.Tasks;

namespace ApiCalls
{
    public class ApiClient
    {
        private readonly HttpClientWrapper _httpClient;
        private readonly JsonSerializerOptions _jsonOptions;

        public ApiClient(HttpClientWrapper httpClient)
        {
            _httpClient = httpClient;
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };
        }

        /// <summary>
        /// Sends a GET request and deserializes the response to type T.
        /// </summary>
        public async Task<T> GetAsync<T>(string endpoint)
        {
            var json = await _httpClient.GetAsync(endpoint);
            return JsonSerializer.Deserialize<T>(json, _jsonOptions);
        }

        /// <summary>
        /// Sends a POST request with serialized data and deserializes the response to type T.
        /// </summary>
        public async Task<T> PostAsync<T>(string endpoint, object data)
        {
            var json = JsonSerializer.Serialize(data);
            var response = await _httpClient.PostAsync(endpoint, json);
            return JsonSerializer.Deserialize<T>(response, _jsonOptions);
        }

        /// <summary>
        /// Sends a POST request with serialized data without expecting a typed response.
        /// </summary>
        public async Task PostAsync(string endpoint, object data)
        {
            var json = JsonSerializer.Serialize(data);
            await _httpClient.PostAsync(endpoint, json);
        }

        /// <summary>
        /// Sends a PUT request with serialized data.
        /// </summary>
        public async Task PutAsync(string endpoint, object data)
        {
            var json = JsonSerializer.Serialize(data);
            await _httpClient.PutAsync(endpoint, json);
        }

        /// <summary>
        /// Sends a DELETE request.
        /// </summary>
        public async Task DeleteAsync(string endpoint)
        {
            await _httpClient.DeleteAsync(endpoint);
        }
    }
}