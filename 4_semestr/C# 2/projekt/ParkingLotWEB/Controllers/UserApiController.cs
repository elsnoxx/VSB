using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;
using Microsoft.AspNetCore.Authorization;
using System.Text.Json;
using BCrypt.Net;
using System.Security.Claims;

[ApiController]
[Route("api/[controller]")]
public class UserApiController : ControllerBase
{
    private readonly UserRepository _repo;

    public UserApiController(UserRepository repo)
    {
        _repo = repo;
    }

    [HttpGet]
    public async Task<IActionResult> GetAll()
    {
        var users = await _repo.GetAllAsync();
        if (users == null || !users.Any())
            return NotFound();
        return Ok(users);
    }

    [HttpGet("{id}")]
    public async Task<IActionResult> Get(int id)
    {
        var user = await _repo.GetByIdAsync(id);
        if (user == null) return NotFound();
        return Ok(user);
    }

    [HttpPut("{id}")]
    public async Task<IActionResult> Update(int id, [FromBody] User model)
    {
        if (string.IsNullOrEmpty(model.Password))
        {
            var original = await _repo.GetByIdAsync(id);
            if (original == null)
                return NotFound();
            model.Password = original.Password;
        }
        else
        {
            model.Password = BCrypt.Net.BCrypt.HashPassword(model.Password);
        }

        var affected = await _repo.UpdateAsync(model);
        if (affected == 0)
            return NotFound();
        return NoContent();
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] User user)
    {
        if (!ModelState.IsValid)
            return BadRequest(ModelState);

        // Hashování hesla před uložením
        user.Password = BCrypt.Net.BCrypt.HashPassword(user.Password);

        var affected = await _repo.CreateAsync(user);
        if (affected == 0)
            return BadRequest();
        return Ok();
    }

    [HttpDelete("{id}")]
    public async Task<IActionResult> Delete(int id)
    {
        var affected = await _repo.DeleteAsync(id);
        if (affected == 0) return NotFound();
        return NoContent();
    }

    [HttpPut("{id}/role")]
    public async Task<IActionResult> ChangeRole(int id, [FromBody] JsonElement body)
    {
        // Directly read the JSON element
        if (body.TryGetProperty("Role", out var roleElement) && roleElement.ValueKind == JsonValueKind.String)
        {
            string role = roleElement.GetString() ?? "";
            
            if (string.IsNullOrEmpty(role))
            {
                return BadRequest(new { error = "Role is required" });
            }

            var affected = await _repo.ChangeRoleAsync(id, role);
            if (affected == 0)
                return NotFound();

            return NoContent();
        }
        
        return BadRequest(new { error = "Invalid request format. Role property is required." });
    }

    [HttpPut("{id}/password")]
    public async Task<IActionResult> ResetPassword(int id, [FromBody] JsonElement body)
    {
        if (!body.TryGetProperty("Password", out var passwordElement) || passwordElement.ValueKind != JsonValueKind.String)
            return BadRequest(new { error = "Password is required" });

        var password = passwordElement.GetString();
        if (string.IsNullOrEmpty(password))
            return BadRequest(new { error = "Password cannot be empty" });

        var affected = await _repo.ResetPasswordAsync(id, password);
        if (affected == 0)
            return NotFound();
        return NoContent();
    }

    [HttpGet("{id}/profile")]
    public async Task<IActionResult> GetProfile(int id)
    {
        var user = await _repo.GetByIdAsync(id);
        if (user == null) return NotFound();

        // Získej auta a převed je na CarDto
        var cars = await _repo.GetCarsByUserIdAsync(id);
        var carDtos = cars.Select(car => new CarDto
        {
            CarId = car.CarId,
            UserId = car.UserId,
            LicensePlate = car.LicensePlate,
            BrandModel = car.BrandModel,
            Color = car.Color
        }).ToList();

        // Získej historii parkování
        var parkingHistory = await _repo.GetParkingHistoryByUserIdAsync(id);
        var parkingHistoryDtos = parkingHistory.Select(history => new ParkingHistoryDto
        {
            LicensePlate = history.LicensePlate,
            ParkingLotName = history.ParkingLotName,
            ArrivalTime = history.ArrivalTime,
            DepartureTime = history.DepartureTime
        }).ToList();

        // Získej aktuálně zaparkovaná auta
        var currentParking = await _repo.GetCurrentParkingByUserIdAsync(id);
        var currentParkingDtos = currentParking.Select(parking => new CurrentParkingDto
        {
            LicensePlate = parking.LicensePlate,
            ParkingLotId = parking.ParkingLotId,
            ParkingSpaceId = parking.ParkingSpaceId,
            ParkingLotName = parking.ParkingLotName,
            ArrivalTime = parking.ArrivalTime
        }).ToList();

        var profile = new UserProfileViewModel
        {
            Id = user.Id,
            Username = user.Username,
            FirstName = user.FirstName,
            LastName = user.LastName,
            Email = user.Email,
            Cars = carDtos,
            ParkingHistory = parkingHistoryDtos,
            CurrentParking = currentParkingDtos
        };

        return Ok(profile);
    }

    [HttpGet("GetUserCars")]
    public async Task<IActionResult> GetUserCars()
    {
        var userId = int.Parse(User.FindFirst(ClaimTypes.NameIdentifier)?.Value ?? "0");
        if (userId == 0)
            return Unauthorized();

        var cars = await _repo.GetCarsByUserIdAsync(userId);
        var carDtos = cars.Select(car => new
        {
            car.LicensePlate,
            car.BrandModel
        });

        return Ok(carDtos);
    }

    
}