using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Text;

public class ParkingLotController : Controller
{
    private readonly ApiClient _apiClient;

    public ParkingLotController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    public async Task<IActionResult> Index()
    {
        var response = await _apiClient.GetAsync("api/ParkingLotApi");
        var lots = new List<ParkingLot>();
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadAsStringAsync();
            lots = JsonSerializer.Deserialize<List<ParkingLot>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new List<ParkingLot>();
        }
        return View(lots);
    }

    public async Task<IActionResult> Details(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var lot = JsonSerializer.Deserialize<ParkingLot>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(lot);
    }

    [HttpGet]
    public IActionResult Create() => View();

    [HttpPost]
    public async Task<IActionResult> Create(ParkingLot lot)
    {
        if (!ModelState.IsValid) return View(lot);
        var response = await _apiClient.PostAsync("api/ParkingLotApi", lot);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(lot);
    }

    [HttpGet]
    public async Task<IActionResult> Edit(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var lot = JsonSerializer.Deserialize<ParkingLot>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(lot);
    }

    [HttpPost]
    public async Task<IActionResult> Edit(ParkingLot lot)
    {
        if (!ModelState.IsValid) return View(lot);
        var response = await _apiClient.PutAsync($"api/ParkingLotApi/{lot.ParkingLotId}", lot);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(lot);
    }

    public async Task<IActionResult> Delete(int id)
    {
        await _apiClient.DeleteAsync($"api/ParkingLotApi/{id}");
        return RedirectToAction("Index");
    }
}