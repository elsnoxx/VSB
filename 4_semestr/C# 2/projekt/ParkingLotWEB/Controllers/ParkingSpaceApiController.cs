using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;

[ApiController]
[Route("api/[controller]")]
public class ParkingSpaceApiController : ControllerBase
{
    private readonly ParkingSpaceRepository _repo;

    public ParkingSpaceApiController(ParkingSpaceRepository repo)
    {
        _repo = repo;
    }

    [HttpGet("history/{parkingSpaceId}")]
    public async Task<IActionResult> GetHistory(int parkingSpaceId)
    {
        var history = await _repo.GetStatusHistoryAsync(parkingSpaceId);
        return Ok(history);
    }

    [HttpPost("occupy/{parkingSpaceId}")]
    public async Task<IActionResult> Occupy(int parkingSpaceId, [FromBody] OccupyRequest req)
    {
        // 1. Změna stavu místa na "occupied" + zápis do StatusHistory
        await _repo.UpdateStatusAsync(parkingSpaceId, "occupied");

        // 2. Zápis do tabulky Occupancy
        await _repo.InsertOccupancyAsync(parkingSpaceId, req.LicensePlate);

        return Ok();
    }

    [HttpPost("release/{parkingSpaceId}")]
    public async Task<IActionResult> Release(int parkingSpaceId)
    {
        // 1. Změna stavu místa na "available" + zápis do historie
        await _repo.UpdateStatusAsync(parkingSpaceId, "available");

        // 2. Ukončení poslední obsazenosti (nastaví end_time, duration, price)
        await _repo.ReleaseOccupancyAsync(parkingSpaceId);

        return Ok();
    }

    [HttpPost("status/{parkingSpaceId}")]
    public async Task<IActionResult> SetStatus(int parkingSpaceId, [FromBody] SetStatusRequest req)
    {
        await _repo.UpdateStatusAsync(parkingSpaceId, req.Status);
        return Ok();
    }

    [HttpGet("occupancy/{parkingSpaceId}")]
    public async Task<IActionResult> GetOccupancy(int parkingSpaceId)
    {
        var occ = await _repo.GetCurrentOccupancyAsync(parkingSpaceId);
        if (occ == null) return NotFound();
        return Ok(occ);
    }
}


