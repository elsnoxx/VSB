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
        int userId;
        if (car.UserId != 0)
            userId = car.UserId;
        else
            userId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier).Value);

        if (!ModelState.IsValid)
            return View(car);

        var carDto = new CarCreateDto
        {
            UserId = userId,
            LicensePlate = car.LicensePlate,
            BrandModel = car.BrandModel!,
            Color = car.Color!
        };

        // Použití PostAsJsonAsync místo ruční serializace
        var response = await _apiClient.PostAsync("api/CarApi/new", carDto);

        if (response.IsSuccessStatusCode)
            return RedirectToAction("Profil", "User", new { id = userId });

        var error = await response.Content.ReadAsStringAsync();
        Console.WriteLine(error);
        ModelState.AddModelError("", error);
        return View(car);
    }

    [HttpPost]
    public async Task<IActionResult> Delete(int id)
    {
        var response = await _apiClient.DeleteAsync($"api/CarApi/{id}");

        if (response.IsSuccessStatusCode)
            return RedirectToAction("Profil", "User");

        var error = await response.Content.ReadAsStringAsync();
        Console.WriteLine(error);
        ModelState.AddModelError("", "Chyba při mazání auta.");
        return RedirectToAction("Profil", "User");
    }
}