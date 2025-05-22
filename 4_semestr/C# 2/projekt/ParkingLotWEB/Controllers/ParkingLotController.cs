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
                SpaceNumber = ps.SpaceNumber
            }).ToList()
        };

        return View(parkingLotViewModel);
    }

    [HttpPost]
    public async Task<IActionResult> ParkCar(int ParkingLotId, string LicensePlate)
    {
        Console.WriteLine($"ParkingLotId: {ParkingLotId}, LicensePlate: {LicensePlate}");
        var response = await _apiClient.PostAsync($"api/ParkingSpaceApi/occupy/{ParkingLotId}", new { licensePlate = LicensePlate });
        if (response.IsSuccessStatusCode)
            TempData["Success"] = "Auto bylo úspěšně zaparkováno.";
        else
            TempData["Error"] = "Nepodařilo se zaparkovat auto.";

        return RedirectToAction("Details", new { id = ParkingLotId });
    }

    [HttpPost]
    public async Task<IActionResult> ReleaseCar(int ParkingSpaceId, int ParkingLotId)
    {
        Console.WriteLine($"ParkingSpaceId: {ParkingSpaceId}, ParkingLotId: {ParkingLotId}");
        var req = new { ParkingSpaceId = ParkingSpaceId, ParkingLotId = ParkingLotId };
        var response = await _apiClient.PostAsync("api/ParkingSpaceApi/release", req);
        if (response.IsSuccessStatusCode)
            TempData["Success"] = "Místo bylo uvolněno.";
        else
            TempData["Error"] = "Nepodařilo se uvolnit místo.";

        return RedirectToAction("Details", new { id = ParkingLotId });
    }

    [HttpPost]
    public async Task<IActionResult> ChangeStatus(int parkingSpaceId, string status, int parkingLotId)
    {
        // Zjisti aktuální stav místa
        var spaceResp = await _apiClient.GetAsync($"api/ParkingSpaceApi/{parkingSpaceId}");
        if (!spaceResp.IsSuccessStatusCode)
        {
            TempData["Error"] = "Nepodařilo se načíst místo.";
            return RedirectToAction("Admin", new { id = parkingLotId });
        }
        var json = await spaceResp.Content.ReadAsStringAsync();
        var space = JsonSerializer.Deserialize<ParkingSpaceViewModel>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (space.Status == "occupied" && status == "unavailable")
        {
            TempData["Error"] = "Nelze označit obsazené místo jako nedostupné!";
            return RedirectToAction("Admin", new { id = parkingLotId });
        }

        await _apiClient.PostAsync($"api/ParkingSpaceApi/status/{parkingSpaceId}", new { status });
        TempData["Success"] = $"Stav místa {parkingSpaceId} byl změněn na {status}.";

        return RedirectToAction("Admin", new { id = parkingLotId });
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Statistics()
    {
        var response = await _apiClient.GetAsync("api/ParkingLotApi/statistics/completed-last-month");
        if (!response.IsSuccessStatusCode)
            return NotFound();

        var json = await response.Content.ReadAsStringAsync();
        var viewModel = JsonSerializer.Deserialize<List<ParkingLotStatisticsViewModel>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return View(viewModel);
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> Charts(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingLotApi/occupancy-timeline/{id}");
        if (!response.IsSuccessStatusCode)
            return NotFound();

        var json = await response.Content.ReadAsStringAsync();
        var data = JsonSerializer.Deserialize<List<OccupancyPointDto>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        ViewBag.ParkingLotId = id;
        return View(data);
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> EditPrice(int parkingLotId, decimal pricePerHour)
    {
        var response = await _apiClient.PutAsync($"api/ParkingLotApi/{parkingLotId}/price", new { pricePerHour });
        if (response.IsSuccessStatusCode)
            return RedirectToAction("Index");
        // případně zobrazit chybu
        return RedirectToAction("Index");
    }

    [Authorize(Roles = "Admin")]
    public async Task<IActionResult> SpaceHistory(int parkingSpaceId)
    {
        // Získání historie pro parkovací místo
        var historyResponse = await _apiClient.GetAsync($"api/ParkingSpaceApi/history/{parkingSpaceId}");
        if (!historyResponse.IsSuccessStatusCode)
            return NotFound();

        var historyJson = await historyResponse.Content.ReadAsStringAsync();
        var historyData = JsonSerializer.Deserialize<List<StatusHistory>>(historyJson, 
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        // Získání informací o parkovacím místě
        var spaceResponse = await _apiClient.GetAsync($"api/ParkingSpaceApi/{parkingSpaceId}");
        if (!spaceResponse.IsSuccessStatusCode)
            return NotFound();

        var spaceJson = await spaceResponse.Content.ReadAsStringAsync();
        var spaceData = JsonSerializer.Deserialize<dynamic>(spaceJson, 
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        // Informace o parkovacím místě pro zobrazení
        ViewBag.SpaceNumber = historyData?.FirstOrDefault()?.SpaceNumber ?? 0;
        ViewBag.ParkingSpaceId = parkingSpaceId;
        ViewBag.ParkingLotId = historyData?.FirstOrDefault()?.ParkingLotId ?? 0;

        return View(historyData);
    }
}