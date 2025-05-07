using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;
using ParkingLotWEB.Models.Entities;
using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Authorization;


[Authorize]
public class ParkingLotController : Controller
{
    private readonly ApiClient _apiClient;

    public ParkingLotController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }
    [Authorize(Roles = "Admin, User")]
    public async Task<IActionResult> Index()
    {
        var response = await _apiClient.GetAsync("api/ParkingLotApi/withFreespaces");
        var lots = new List<ParkingLot>();
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadAsStringAsync();
            lots = JsonSerializer.Deserialize<List<ParkingLot>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new List<ParkingLot>();
        }
        return View(lots);
    }

    [Authorize(Roles = "Admin, User")]
    public async Task<IActionResult> Details(int id)
    {
        var userId = User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/details/{id}/user/{userId}");

        if (!response.IsSuccessStatusCode) return NotFound();

        var json = await response.Content.ReadAsStringAsync();
        var lot = JsonSerializer.Deserialize<ParkingLotDetailsViewModel>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (lot == null) return NotFound();

        return View(lot);
    }

    [HttpGet]
    [Authorize(Roles = "Admin, User")]
    public IActionResult Create() => View();

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Create(ParkingLotCreateViewModel model)
    {
        if (!ModelState.IsValid) return View(model);

        var lot = new ParkingLot
        {
            Name = model.Name,
            Latitude = model.Latitude,
            Longitude = model.Longitude,
            Capacity = model.Capacity
        };

        var response = await _apiClient.PostAsync("api/ParkingLotApi", lot);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(model);
    }

    [HttpGet]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Edit(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/{id}");
        if (!response.IsSuccessStatusCode) return NotFound();
        var json = await response.Content.ReadAsStringAsync();
        var lot = JsonSerializer.Deserialize<ParkingLot>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(lot);
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Edit(ParkingLot lot)
    {
        if (!ModelState.IsValid) return View(lot);
        var response = await _apiClient.PutAsync($"api/ParkingLotApi/{lot.ParkingLotId}", lot);
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        return View(lot);
    }
    
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Delete(int id)
    {
        await _apiClient.DeleteAsync($"api/ParkingLotApi/{id}");
        return RedirectToAction("Index");
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Admin(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/{id}");
        if (!response.IsSuccessStatusCode)
        {
            return NotFound();
        }

        var json = await response.Content.ReadAsStringAsync();
        var parkingLot = JsonSerializer.Deserialize<ParkingLotDto>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        
        if (parkingLot == null)
        {
            return NotFound();
        }

        var parkingLotViewModel = new ParkingLotViewModel
        {
            ParkingLotId = parkingLot.ParkingLotId,
            Name = parkingLot.Name,
            Latitude = (double)parkingLot.Latitude,
            Longitude = (double)parkingLot.Longitude,
            Capacity = parkingLot.Capacity,
            FreeSpaces = parkingLot.Capacity - parkingLot.ParkingSpaces?.Count(ps => ps.Status == "occupied") ?? 0,
            ParkingSpaces = parkingLot.ParkingSpaces?.Select(ps => new ParkingSpaceViewModel
            {
                ParkingSpaceId = ps.ParkingSpaceId,
                Status = ps.Status,
                SpaceNumber = ps.SpaceNumber // Předpokládám, že tato vlastnost existuje v ParkingSpaceDto
            }).ToList()
        };

        return View(parkingLotViewModel);
    }
}