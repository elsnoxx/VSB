using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using Microsoft.AspNetCore.Authorization;

[ApiController]
[Route("api/[controller]")]
public class UserApiController : ControllerBase
{
    private readonly UserRepository _repo;

    public UserApiController(IConfiguration config)
    {
        _repo = new UserRepository(config);
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
        using var conn = _repo.CreateConnection();
        string sql;
        if (!string.IsNullOrEmpty(model.Password))
        {
            sql = "UPDATE Users SET Username = @Username, Password = @Password, Role = @Role WHERE Id = @Id";
            await conn.ExecuteAsync(sql, new { model.Username, model.Password, model.Role, Id = id });
        }
        else
        {
            sql = "UPDATE Users SET Username = @Username, Role = @Role WHERE Id = @Id";
            await conn.ExecuteAsync(sql, new { model.Username, model.Role, Id = id });
        }
        return NoContent();
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] User user)
    {
        if (!ModelState.IsValid)
            return BadRequest(ModelState);

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
}