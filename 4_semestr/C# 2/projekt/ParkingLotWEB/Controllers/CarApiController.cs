using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;

[ApiController]
[Route("api/[controller]")]
public class CarApiController : ControllerBase
{
    private readonly CarRepository _repo;

    public CarApiController(CarRepository repo)
    {
        _repo = repo;
    }

    [HttpPost("new")]
    public async Task<IActionResult> Create([FromBody] Car car)
    {
        await _repo.CreateAsync(car);
        return Ok();
    }

    [HttpGet("user/{userId}")]
    public async Task<IActionResult> GetByUserId(int userId)
    {
        var cars = await _repo.GetByUserIdAsync(userId);
        return Ok(cars);
    }
}