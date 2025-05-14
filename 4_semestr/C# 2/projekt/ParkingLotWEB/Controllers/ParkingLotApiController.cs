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
            ParkingSpaces = spaces.Select(s => new ParkingSpaceDto
            {
                ParkingSpaceId = s.ParkingSpaceId,
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
            FreeSpaces = lot.FreeSpaces,
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
}