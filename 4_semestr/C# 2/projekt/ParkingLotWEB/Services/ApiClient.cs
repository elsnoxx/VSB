using System.Net.Http.Headers;
using System.Text;
using Microsoft.AspNetCore.Http;

public class ApiClient
{
    private readonly HttpClient _client;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly string _apiKey;

    public ApiClient(HttpClient client, IHttpContextAccessor httpContextAccessor, IConfiguration configuration)
    {
        _client = client;
        _httpContextAccessor = httpContextAccessor;
        _apiKey = configuration["API-KEY:Key"];
    }

    private void AddApiKeyHeader()
    {
        if (!_client.DefaultRequestHeaders.Contains("X-Api-Key"))
            _client.DefaultRequestHeaders.Add("X-Api-Key", _apiKey);
    }

    private void AddJwtHeader()
    {
        var token = _httpContextAccessor.HttpContext?.Request.Cookies["jwt"];
        if (!string.IsNullOrEmpty(token))
            _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        AddApiKeyHeader();
    }

    public async Task<HttpResponseMessage> GetAsync(string url)
    {
        AddJwtHeader();
        return await _client.GetAsync(url);
    }

    public async Task<HttpResponseMessage> PostAsync(string url, object data)
    {
        AddJwtHeader();
        var json = System.Text.Json.JsonSerializer.Serialize(data);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        return await _client.PostAsync(url, content);
    }

    public async Task<HttpResponseMessage> PutAsync(string url, object data)
    {
        AddJwtHeader();
        var json = System.Text.Json.JsonSerializer.Serialize(data);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        return await _client.PutAsync(url, content);
    }

    public async Task<HttpResponseMessage> DeleteAsync(string url)
    {
        AddJwtHeader();
        return await _client.DeleteAsync(url);
    }
}