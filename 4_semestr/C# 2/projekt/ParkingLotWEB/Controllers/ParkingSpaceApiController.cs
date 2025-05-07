using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;

[ApiController]
[Route("api/[controller]")]
public class ParkingSpaceApiController : ControllerBase
{
    private readonly ParkingSpaceRepository _repo;
    private readonly ParkingLotRepository _parkingLotRepo;

    public ParkingSpaceApiController(ParkingSpaceRepository repo, ParkingLotRepository parkingLotRepo)
    {
        _repo = repo;
        _parkingLotRepo = parkingLotRepo;
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

    [HttpPost("release")]
    public async Task<IActionResult> Release([FromBody] ReleaseRequest req)
    {
        // 1. Změna stavu místa na "available" + zápis do historie
        await _repo.UpdateStatusAsync(req.ParkingSpaceId, "available");

        // 2. Ukončení poslední obsazenosti (nastaví end_time, duration, price)
        await _repo.ReleaseOccupancyAsync(req.ParkingSpaceId, req.ParkingLotId);

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

    [HttpGet("{id}")]
    public async Task<IActionResult> Get(int id)
    {
        var space = await _repo.GetByIdAsync(id);
        if (space == null) return NotFound();

        // Pokud potřebujete další informace o tomto parkovacím místě
        var details = await _repo.GetParkingSpaceDetailsAsync(id);
        
        // Vytvořte nový objekt s kompletními informacemi
        var result = new
        {
            ParkingSpaceId = space.ParkingSpaceId,
            ParkingLotId = space.ParkingLotId,
            Status = space.Status,
            // Další vlastnosti, které potřebujete...
            Details = details
        };

        return Ok(result);
    }

    [HttpGet("lot/{parkingLotId}")]
    public async Task<IActionResult> GetByLotId(int parkingLotId)
    {
        var spaces = await _repo.GetSpacesWithOwnerAsync(parkingLotId);
        if (spaces == null || !spaces.Any()) return NotFound();
        
        // Získejte informace o parkovišti
        var parkingLot = await _parkingLotRepo.GetByIdAsync(parkingLotId);
        if (parkingLot == null) return NotFound();

        // Vytvořte kombinovaný DTO objekt
        var result = new
        {
            ParkingLotId = parkingLot.ParkingLotId,
            Name = parkingLot.Name,
            // Další vlastnosti parkoviště...
            ParkingSpaces = spaces.ToList()
        };

        return Ok(result);
    }
}


