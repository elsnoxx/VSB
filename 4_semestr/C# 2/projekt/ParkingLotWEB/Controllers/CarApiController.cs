using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;
using System.Security.Claims;
using System.Text.Json;

[ApiController]
[Route("api/[controller]")]
public class CarApiController : ControllerBase
{
    private readonly CarRepository _repo;
    private readonly UserRepository _userRepo;

    public CarApiController(CarRepository repo, UserRepository userRepo)
    {
        _repo = repo;
        _userRepo = userRepo;
    }

    [HttpPost("new")]
    public async Task<IActionResult> Create([FromBody] CarCreateDto carDto)
    {
        using var reader = new StreamReader(Request.Body);
        var rawJson = await reader.ReadToEndAsync();
        Console.WriteLine($"Raw JSON received: {rawJson}");

        Console.WriteLine($"Deserialized JSON: {JsonSerializer.Serialize(carDto)}");

        if (string.IsNullOrWhiteSpace(carDto.LicensePlate) ||
            string.IsNullOrWhiteSpace(carDto.BrandModel) ||
            string.IsNullOrWhiteSpace(carDto.Color))
        {
            return BadRequest("Všechna pole musí být vyplněna.");
        }

        if (carDto.UserId == 0)
        {
            carDto.UserId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier).Value);
        }

        var car = new Car
        {
            UserId = carDto.UserId,
            LicensePlate = carDto.LicensePlate,
            BrandModel = carDto.BrandModel,
            Color = carDto.Color
        };

        await _repo.CreateAsync(car);
        return Ok();
    }

    [HttpGet("user/{userId}")]
    public async Task<IActionResult> GetByUserId(int userId)
    {
        var cars = await _repo.GetByUserIdAsync(userId);
        return Ok(cars);
    }
    
    [HttpGet("GetUserCars/{userId}")]
    public async Task<IActionResult> GetUserCars(int userId)
    {
        var cars = await _userRepo.GetCarsByUserIdAsync(userId);
        return Ok(cars.Select(car => new { car.LicensePlate, car.BrandModel }));
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> Delete(int id)
    {
        var affectedRows = await _repo.DeleteAsync(id);

        if (affectedRows == 0)
            return NotFound("Auto nebylo nalezeno.");

        return NoContent();
    }
}