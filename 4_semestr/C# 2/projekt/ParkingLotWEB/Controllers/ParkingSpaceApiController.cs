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
    private Random _random = new Random();
    private const int MIN_PARKING_SPACE_ID = 1;

    public ParkingSpaceApiController(ParkingSpaceRepository repo, ParkingLotRepository parkingLotRepo)
    {
        _repo = repo;
        _parkingLotRepo = parkingLotRepo;
    }

    [HttpGet("history/{parkingSpaceId}")]
    [ProducesResponseType(typeof(IEnumerable<StatusHistory>), 200)]
    public async Task<IActionResult> GetHistory(int parkingSpaceId)
    {
        var history = await _repo.GetStatusHistoryAsync(parkingSpaceId);
        return Ok(history);
    }

    [HttpPost("occupy/{parkingLotId}")]
    [ProducesResponseType(typeof(IEnumerable<Occupancy>), 200)]
    public async Task<IActionResult> Occupy(int parkingLotId, [FromBody] OccupyRequest req)
    {
        var freeSpaces = (await _repo.GetAvailableSpacesAsync(parkingLotId)).ToList();
        Console.WriteLine($"Free spaces count: {freeSpaces.Count}");
        if (!freeSpaces.Any())
            return BadRequest("No free spaces available.");

        var selected = freeSpaces[_random.Next(freeSpaces.Count)];
        Console.WriteLine($"Selected space: {selected.ParkingSpaceId}");

        await _repo.UpdateStatusAsync(selected.ParkingSpaceId, "occupied");
        await _repo.InsertOccupancyAsync(selected.ParkingSpaceId, req.LicensePlate);
        await _repo.InsertStatusHistoryAsync(selected.ParkingSpaceId, "occupied");

        return Ok(new { ParkingSpaceId = selected.ParkingSpaceId, SpaceNumber = selected.SpaceNumber });
    }

    [HttpPost("release")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Release([FromBody] ReleaseRequest req)
    {
        // 1. Změna stavu místa na "available" + zápis do historie
        await _repo.UpdateStatusAsync(req.ParkingSpaceId, "available");

        // 2. Ukončení poslední obsazenosti
        await _repo.ReleaseOccupancyAsync(req.ParkingSpaceId, req.ParkingLotId);

        await _repo.InsertStatusHistoryAsync(req.ParkingSpaceId, "available");

        return Ok();
    }

    [HttpPost("status/{parkingSpaceId}")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> SetStatus(int parkingSpaceId, [FromBody] SetStatusRequest req)
    {
        Console.WriteLine($"Setting status for space {parkingSpaceId} to {req.Status}");
        await _repo.UpdateStatusAsync(parkingSpaceId, req.Status);
        await _repo.InsertStatusHistoryAsync(parkingSpaceId, req.Status);
        return Ok();
    }

    [HttpGet("occupancy/{parkingSpaceId}")]
    [ProducesResponseType(typeof(Occupancy), 200)]
    public async Task<IActionResult> GetOccupancy(int parkingSpaceId)
    {
        var occ = await _repo.GetCurrentOccupancyAsync(parkingSpaceId);
        if (occ == null) return NotFound();
        return Ok(occ);
    }

    [HttpGet("{id}")]
    [ProducesResponseType(typeof(ParkingSpace), 200)]
    public async Task<IActionResult> Get(int id)
    {
        var space = await _repo.GetByIdAsync(id);
        if (space == null) return NotFound();

        var details = await _repo.GetParkingSpaceDetailsAsync(id);

        var result = new
        {
            ParkingSpaceId = space.ParkingSpaceId,
            ParkingLotId = space.ParkingLotId,
            Status = space.Status,
            Details = details
        };

        return Ok(result);
    }

    [HttpGet("lot/{parkingLotId}")]
    [ProducesResponseType(typeof(IEnumerable<ParkingSpace>), 200)]
    public async Task<IActionResult> GetByLotId(int parkingLotId)
    {
        var spaces = await _repo.GetSpacesWithOwnerAsync(parkingLotId);
        if (spaces == null || !spaces.Any()) return NotFound();

        var parkingLot = await _parkingLotRepo.GetByIdAsync(parkingLotId);
        if (parkingLot == null) return NotFound();

        var result = new
        {
            ParkingLotId = parkingLot.ParkingLotId,
            Name = parkingLot.Name,
            ParkingSpaces = spaces.ToList()
        };

        return Ok(result);
    }

    
}


