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
    public async Task<IActionResult> GetAll() => Ok(await _repo.GetAllAsync());

    [HttpGet("withFreespaces")]
    public async Task<IActionResult> GetAllWithFreeSpaces()
    {
        var lots = await _repo.GetAllWithFreeSpacesAsync();
        return Ok(lots);
    }

    [HttpGet("{id}")]
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
    public async Task<IActionResult> Create(ParkingLot lot)
    {
        await _repo.CreateAsync(lot);
        return CreatedAtAction(nameof(Get), new { id = lot.ParkingLotId }, lot);
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> Update(int id, ParkingLot lot)
    {
        if (id != lot.ParkingLotId) return BadRequest();
        await _repo.UpdateAsync(lot);
        return NoContent();
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> Delete(int id)
    {
        await _repo.DeleteAsync(id);
        return NoContent();
    }

    [HttpGet("details/{lotId}/user/{userId}")]
    public async Task<IActionResult> GetParkingLotDetailsWithUserCars(int lotId, int userId)
    {
        var lot = await _repo.GetByIdAsync(lotId);

        // Získej parkovací místa včetně vlastníků
        var spacesWithOwner = await _spaceRepo.GetSpacesWithOwnerAsync(lotId);

        // Získání aut uživatele
        var userCars = await _userRepo.GetCarsByUserIdAsync(userId);

        var occupiedPlates = await _spaceRepo.GetAllOccupiedLicensePlatesAsync();

        var details = new ParkingLotDetailsViewModel
        {
            ParkingLotId = lot.ParkingLotId,
            Name = lot.Name,
            Latitude = lot.Latitude,
            Longitude = lot.Longitude,
            Capacity = lot.Capacity,
            FreeSpaces = lot.Capacity - lot.FreeSpaces,
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
    public async Task<IActionResult> GetOccupancyTimeline(int id)
    {
        var data = await _repo.GetOccupancyTimelineAsync(id);
        return Ok(data);
    }

    [HttpPut("{id}/price")]
    public async Task<IActionResult> UpdatePrice(int id, [FromBody] System.Text.Json.JsonElement body)
    {
        if (!body.TryGetProperty("pricePerHour", out var priceProp) || !priceProp.TryGetDecimal(out var pricePerHour))
            return BadRequest("Missing or invalid pricePerHour.");

        var result = await _repo.UpdatePricePerHourAsync(id, pricePerHour);
        if (result > 0)
            return Ok();
        return NotFound();
    }
}