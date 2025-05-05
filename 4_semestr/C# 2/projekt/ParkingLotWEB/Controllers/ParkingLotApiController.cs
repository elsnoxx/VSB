using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using ParkingLotWEB.Database;

[ApiController]
[Route("api/[controller]")]
public class ParkingLotApiController : ControllerBase
{
    private readonly ParkingLotRepository _repo;

    public ParkingLotApiController(ParkingLotRepository repo)
    {
        _repo = repo;
    }

    [HttpGet]
    public async Task<IActionResult> GetAll() => Ok(await _repo.GetAllAsync());

    [HttpGet("{id}")]
    public async Task<IActionResult> Get(int id)
    {
        var lot = await _repo.GetByIdAsync(id);
        if (lot == null) return NotFound();
        return Ok(lot);
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