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
    

    public ParkingLotApiController(ParkingLotRepository repo, ParkingSpaceRepository spaceRepo)
    {
        _repo = repo;
        _spaceRepo = spaceRepo;
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
}