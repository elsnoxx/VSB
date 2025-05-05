using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Authorization;

// [Authorize(Roles = "Admin")]
public class UserController : Controller
{
    private readonly IHttpClientFactory _clientFactory;

    public UserController(IHttpClientFactory clientFactory)
    {
        _clientFactory = clientFactory;
    }

    private HttpClient CreateApiClient()
    {
        var client = _clientFactory.CreateClient();
        var token = HttpContext.Request.Cookies["jwt"] ?? ""; // nebo localStorage přes JS
        if (!string.IsNullOrEmpty(token))
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        client.BaseAddress = new Uri("http://localhost:5062/"); // uprav dle svého
        return client;
    }

    public async Task<IActionResult> Index()
    {
        var client = CreateApiClient();
        var response = await client.GetAsync("api/UserApi");
        var users = new List<User>();
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadAsStringAsync();
            users = JsonSerializer.Deserialize<List<User>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new List<User>();
        }
        return View(users);
    }

    public async Task<IActionResult> Edit(int id)
    {
        var client = CreateApiClient();
        var response = await client.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    public async Task<IActionResult> Edit(User user)
    {
        var client = CreateApiClient();
        var json = JsonSerializer.Serialize(user);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await client.PutAsync($"api/UserApi/{user.Id}", content);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(user);
    }

    public async Task<IActionResult> Create()
    {
        return View(new User());
    }

    [HttpPost]
    public async Task<IActionResult> Create(User user)
    {
        var client = CreateApiClient();
        var json = JsonSerializer.Serialize(user);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await client.PostAsync("api/UserApi", content);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(user);
    }

    public async Task<IActionResult> Delete(int id)
    {
        var client = CreateApiClient();
        var response = await client.DeleteAsync($"api/UserApi/{id}");
        return RedirectToAction("Index");
    }
}