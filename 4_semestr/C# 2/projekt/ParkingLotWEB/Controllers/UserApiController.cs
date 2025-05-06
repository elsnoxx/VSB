using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using Microsoft.AspNetCore.Authorization;
using System.Text.Json;
using BCrypt.Net;

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
        var affected = await _repo.UpdateAsync(id, model);
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
    public async Task<IActionResult> ResetPassword(int id, [FromBody] User model)
    {
        // Hashování nového hesla před uložením
        var hashedPassword = BCrypt.Net.BCrypt.HashPassword(model.Password);

        var affected = await _repo.ResetPasswordAsync(id, hashedPassword);
        if (affected == 0)
            return NotFound();
        return NoContent();
    }

    [HttpGet("{id}/profile")]
    public async Task<IActionResult> GetProfile(int id)
    {
        var user = await _repo.GetByIdAsync(id);
        if (user == null) return NotFound();

        // Zde načti i auta uživatele (příklad)
        var cars = await _repo.GetCarsByUserIdAsync(id);

        var profile = new UserProfileViewModel
        {
            Id = user.Id,
            Username = user.Username,
            FirstName = user.FirstName,
            LastName = user.LastName,
            Email = user.Email,
            Cars = cars
        };

        return Ok(profile);
    }
}