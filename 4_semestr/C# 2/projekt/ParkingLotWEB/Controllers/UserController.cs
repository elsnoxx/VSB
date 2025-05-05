using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Authorization;

// [Authorize(Roles = "Admin")]
public class UserController : Controller
{
    private readonly ApiClient _apiClient;

    public UserController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    public async Task<IActionResult> Index()
    {
        var response = await _apiClient.GetAsync("api/UserApi");
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
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    public async Task<IActionResult> Edit(User user)
    {
        var json = JsonSerializer.Serialize(user);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _apiClient.PutAsync($"api/UserApi/{user.Id}", content);
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
        var json = JsonSerializer.Serialize(user);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _apiClient.PostAsync("api/UserApi", content);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(user);
    }

    public async Task<IActionResult> Delete(int id)
    {
        var response = await _apiClient.DeleteAsync($"api/UserApi/{id}");
        return RedirectToAction("Index");
    }

    public async Task<IActionResult> ChangeRole(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    public async Task<IActionResult> ChangeRole(int id, string role)
    {
        var content = new StringContent(JsonSerializer.Serialize(new { Role = role }), Encoding.UTF8, "application/json");
        var response = await _apiClient.PutAsync($"api/UserApi/{id}/role", content);
        if (!response.IsSuccessStatusCode)
        {
            ModelState.AddModelError("", "Změna role se nezdařila – nemáte oprávnění nebo došlo k chybě.");
            var user = new User { Id = id, Role = role };
            return View(user);
        }
        return RedirectToAction("Index");
    }

    public async Task<IActionResult> ResetPassword(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    public async Task<IActionResult> ResetPassword(int id, string password)
    {
        var content = new StringContent(JsonSerializer.Serialize(new { Password = password }), Encoding.UTF8, "application/json");
        var response = await _apiClient.PutAsync($"api/UserApi/{id}/password", content);
        return RedirectToAction("Index");
    }
}