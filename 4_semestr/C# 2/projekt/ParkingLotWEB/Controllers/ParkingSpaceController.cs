using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;

[Authorize(Roles = "Admin, User")]
public class ParkingSpaceController : Controller
{
    private readonly ApiClient _apiClient;
    public ParkingSpaceController(ApiClient apiClient)
    {
        _apiClient = apiClient;
    }

    public async Task<IActionResult> OccupancyDetail(int id)
    {
        var response = await _apiClient.GetAsync($"api/ParkingSpaceApi/occupancy/{id}");
        if (!response.IsSuccessStatusCode)
            return NotFound();

        var json = await response.Content.ReadAsStringAsync();
        var occupancy = JsonSerializer.Deserialize<Occupancy>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        if (occupancy == null) return NotFound();
        return View(occupancy);
    }
}