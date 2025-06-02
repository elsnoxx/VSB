using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models.Entities;



[ApiController]
[Route("api/[controller]")]
public class ParkingLotApiController : ControllerBase
{
    private readonly ParkingLotRepository _repo;
    private readonly ParkingSpaceRepository _spaceRepo;
    private readonly UserRepository _userRepo;


    public ParkingLotApiController(ParkingLotRepository repo, ParkingSpaceRepository spaceRepo, UserRepository userRepo)
    {
        _repo = repo;
        _spaceRepo = spaceRepo;
        _userRepo = userRepo;
    }

    [HttpGet("all")]
    [ProducesResponseType(typeof(ParkingLot), 200)]
    public async Task<IActionResult> GetAll() => Ok(await _repo.GetAllAsync());

    [HttpGet("withFreespaces")]
    [ProducesResponseType(typeof(ParkingLot), 200)]
    public async Task<IActionResult> GetAllWithFreeSpaces()
    {
        var lots = await _repo.GetAllWithFreeSpacesAsync();
        
        
        return Ok(lots);
    }

    [HttpGet("{id}")]
    [ProducesResponseType(typeof(ParkingLotDto), 200)]
    public async Task<IActionResult> Get(int id)
    {
        var lot = await _repo.GetByIdAsync(id);
        var spaces = await _spaceRepo.GetSpacesAsync(id);

        var dto = new ParkingLotDto
        {
            ParkingLotId = lot.ParkingLotId,
            Name = lot.Name,
            Latitude = lot.Latitude,
            Longitude = lot.Longitude,
            Capacity = lot.Capacity,
            FreeSpaces = lot.FreeSpaces,
            PricePerHour = lot.PricePerHour,
            ParkingSpaces = spaces.Select(s => new ParkingSpaceDto
            {
                ParkingSpaceId = s.ParkingSpaceId,
                SpaceNumber = s.SpaceNumber,
                ParkingLotId = s.ParkingLotId,
                Status = s.Status
            }).ToList()
        };

        return Ok(dto);
    }

    [HttpPost]
    [ProducesResponseType(typeof(ParkingLot), StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> Create(ParkingLot lot)
    {
        await _repo.CreateAsync(lot);
        return CreatedAtAction(nameof(Get), new { id = lot.ParkingLotId }, lot);
    }

    [HttpPut("{id}")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Update(int id, ParkingLot lot)
    {
        if (id != lot.ParkingLotId) return BadRequest();
        await _repo.UpdateAsync(lot);
        return NoContent();
    }

    [HttpDelete("{id}")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Delete(int id)
    {
        await _repo.DeleteAsync(id);
        return NoContent();
    }

    [HttpGet("details/{lotId}/user/{userId}")]
    [ProducesResponseType(typeof(ParkingLotDetailsViewModel), 200)]
    public async Task<IActionResult> GetParkingLotDetailsWithUserCars(int lotId, int userId)
    {
        var lot = await _repo.GetByIdAsync(lotId);

        // Získej parkovací místa včetně vlastníků
        var spacesWithOwner = await _spaceRepo.GetSpacesWithOwnerAsync(lotId);

        // Získání aut uživatele
        var userCars = await _userRepo.GetCarsByUserIdAsync(userId);

        var occupiedPlates = await _spaceRepo.GetAllOccupiedLicensePlatesAsync();
        Console.WriteLine($"Occupied plates: {lot.FreeSpaces}");

        var details = new ParkingLotDetailsViewModel
        {
            ParkingLotId = lot.ParkingLotId,
            Name = lot.Name,
            Latitude = lot.Latitude,
            Longitude = lot.Longitude,
            Capacity = lot.Capacity,
            FreeSpaces = lot.FreeSpaces,
            PricePerHour = lot.PricePerHour,
            ParkingSpaces = spacesWithOwner.ToList(),
            UserCars = userCars?.Select(car => new CarDto
            {
                LicensePlate = car.LicensePlate,
                BrandModel = car.BrandModel
            }).ToList(),
            ParkingSpacesWithDetails = spacesWithOwner.Select(s => new ParkingSpaceWithDetails
            {
                ParkingSpaceId = s.ParkingSpaceId,
                SpaceNumber = s.SpaceNumber,
                Status = s.Status,
                OwnerId = s.OwnerId
            }).ToList(),
            OccupiedLicensePlates = occupiedPlates.ToList()
        };

        return Ok(details);
    }


    [HttpGet("statistics/completed-last-month")]
    [ProducesResponseType(typeof(IEnumerable<ParkingLotStatisticsViewModel>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetCompletedParkingsLastMonth()
    {
        var stats = await _repo.GetCompletedParkingsLastMonthAsync();
        var lots = await _repo.GetAllAsync();
        var result = lots.Select(lot => new
        {
            lot.ParkingLotId,
            lot.Name,
            CompletedParkings = stats.ContainsKey(lot.ParkingLotId) ? stats[lot.ParkingLotId] : 0
        }).ToList();
        return Ok(result);
    }

    [HttpGet("occupancy-timeline/{id}")]
    [ProducesResponseType(typeof(IEnumerable<OccupancyPointDto>), 200)]
    public async Task<IActionResult> GetOccupancyTimeline(int id)
    {
        var data = await _repo.GetOccupancyTimelineAsync(id);
        return Ok(data);
    }

    [HttpPut("{id}/price")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> UpdatePrice(int id, [FromBody] System.Text.Json.JsonElement body)
    {
        if (!body.TryGetProperty("pricePerHour", out var priceProp) || !priceProp.TryGetDecimal(out var pricePerHour))
            return BadRequest("Missing or invalid pricePerHour.");

        var result = await _repo.UpdatePricePerHourAsync(id, pricePerHour);
        if (result > 0)
            return Ok();
        return NotFound();
    }

    [HttpGet("parkingLotId/{parkingspaceId}")]
    [ProducesResponseType(typeof(ParkingLot), 200)]
    public async Task<IActionResult> GetParkingLotByParkingSpaceId(int parkingspaceId)
    {
        var parkingLot = await _spaceRepo.GetParkingSpaceByParkingLotAsync(parkingspaceId);
        Console.WriteLine($"ParkingLotApiController: ParkingLotId for ParkingSpaceId {parkingLot}");
        if (parkingLot == null) return NotFound();

        return Ok(parkingLot);
    }
}