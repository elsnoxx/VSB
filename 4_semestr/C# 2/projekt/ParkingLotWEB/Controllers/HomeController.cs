using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Text;
using System.Diagnostics;

namespace ParkingLotWEB.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly ApiClient _apiClient;

    public HomeController(ILogger<HomeController> logger, ApiClient apiClient)
    {
        _logger = logger;
        _apiClient = apiClient;
    }

    public async Task<IActionResult> Index()
    {
        // Získání všech parkovišť z API
        var response = await _apiClient.GetAsync("api/ParkingLotApi");
        var lots = new List<ParkingLot>();
        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadAsStringAsync();
            lots = JsonSerializer.Deserialize<List<ParkingLot>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new List<ParkingLot>();
        }
        return View(lots);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
