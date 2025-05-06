using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Security.Claims;
using System.Text.Json;
using System.Text;

public class CarController : Controller
{
    private readonly ApiClient _apiClient;

    public CarController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    [HttpGet]
    public IActionResult Create() => View();

    [HttpPost]
    public async Task<IActionResult> Create(Car car)
    {
        var userId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)!.Value);
        car.UserId = userId;

        var json = JsonSerializer.Serialize(car);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _apiClient.PostAsync("api/CarApi/new", content);

        if (response.IsSuccessStatusCode)
            return RedirectToAction("Profil", "User", new { id = userId });

        // Výpis chybové odpovědi z API
        var error = await response.Content.ReadAsStringAsync();
        Console.WriteLine(error); // Přidej tento řádek pro výpis do konzole
        ModelState.AddModelError("", error);
        return View(car);
    }
}