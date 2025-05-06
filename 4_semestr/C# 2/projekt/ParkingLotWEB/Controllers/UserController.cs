using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Authorization;

[Authorize(Roles = "Admin")]
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
        if (!ModelState.IsValid)
            return View(user);

        var response = await _apiClient.PostAsync("api/UserApi", user);
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
    public async Task<IActionResult> ChangeRole(int id, RoleChange model)
    {
        if (string.IsNullOrEmpty(model.Role))
            return BadRequest();

        // Create JSON payload with Role property
        var jsonPayload = JsonSerializer.Serialize(new { Role = model.Role });
        Console.WriteLine($"Sending JSON: {jsonPayload}");
        
        var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
        // Use PutAsync to match the API's PUT endpoint
        var response = await _apiClient.PutAsync($"api/UserApi/{id}/role", content);
        
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            Console.WriteLine($"Error response: {errorContent}, Status: {response.StatusCode}");
        }
        
        // To show the view again, we need user data
        var userResponse = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (userResponse.IsSuccessStatusCode)
        {
            var json = await userResponse.Content.ReadAsStringAsync();
            var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
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

    [Authorize(Roles = "Admin, User")]
    public async Task<IActionResult> Profil(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}/profile");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<UserProfileViewModel>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }
}