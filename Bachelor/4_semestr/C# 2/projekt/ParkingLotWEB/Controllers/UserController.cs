using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Authorization;

[Authorize]
public class UserController : Controller
{
    private readonly ApiClient _apiClient;

    public UserController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    [Authorize(Roles = "Admin")]
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

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Edit(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Edit(User user)
    {
        // Pokud je Password null, nastav na prázdný string (aby prošla validace v API)
        if (user.Password == null)
            user.Password = "";

        if (string.IsNullOrEmpty(user.Password))
        {
            var response = await _apiClient.GetAsync($"api/UserApi/{user.Id}");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                var original = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                if (original != null)
                    user.Password = original.Password;
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(user));
        var responsePut = await _apiClient.PutAsync($"api/UserApi/{user.Id}", user);
        if (responsePut.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(user);
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Create()
    {
        return View(new User());
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Create(User user)
    {
        if (!ModelState.IsValid)
            return View(user);

        var response = await _apiClient.PostAsync("api/UserApi", user);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(user);
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Delete(int id)
    {
        var response = await _apiClient.DeleteAsync($"api/UserApi/{id}");
        return RedirectToAction("Index");
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> ChangeRole(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(user);
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> ChangeRole(int id, RoleChange model)
    {
        if (string.IsNullOrEmpty(model.Role))
            return BadRequest();

        // Předávej pouze objekt, ne serializovaný JSON!
        var response = await _apiClient.PutAsync($"api/UserApi/{id}/role", new { Role = model.Role });

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

    // GET
    [Authorize(Roles = "Admin,User")]
    public async Task<IActionResult> ResetPassword(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        var model = new ResetPassword { Id = id };
        ViewBag.User = user;
        return View(model);
    }

    // POST
    [HttpPost]
    [Authorize(Roles = "Admin,User")]
    public async Task<IActionResult> ResetPassword(ResetPassword model)
    {
        if (!ModelState.IsValid)
        {
            // Znovu načti uživatele pro ViewBag
            var response = await _apiClient.GetAsync($"api/UserApi/{model.Id}");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                ViewBag.User = user;
            }
            return View(model);
        }

        // Pošli nové heslo na API
        var responsePut = await _apiClient.PutAsync($"api/UserApi/{model.Id}/password", new { Password = model.Password });
        if (responsePut.IsSuccessStatusCode)
            return RedirectToAction("Index");

        // Chyba – znovu načti uživatele
        var resp = await _apiClient.GetAsync($"api/UserApi/{model.Id}");
        if (resp.IsSuccessStatusCode)
        {
            var json = await resp.Content.ReadAsStringAsync();
            var user = JsonSerializer.Deserialize<User>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            ViewBag.User = user;
        }
        return View(model);
    }

    [Authorize(Roles = "Admin, User")]
    public async Task<IActionResult> Profil(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}/profile");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var user = JsonSerializer.Deserialize<UserProfileViewModel>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        System.Console.WriteLine($"User: {user.FirstName} {user.LastName}, Email: {user.Email} {user.Email}");
        return View(user);
    }

    [Authorize(Roles = "Admin, User")]
    [HttpGet("GetUserCars/{id}")]
    public async Task<IActionResult> GetUserCars(int id)
    {
        var response = await _apiClient.GetAsync($"api/UserApi/{id}/cars");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var cars = JsonSerializer.Deserialize<List<Car>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new List<Car>();
        return PartialView("_UserCars", cars);
    }
}