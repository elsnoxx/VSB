using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Dapper;
using ParkingLotWEB.Database;
using ParkingLotWEB.Models;
using System.Net.Http.Headers;

[ApiController]
[Route("api/[controller]")]
public class AuthApiController : ControllerBase
{
    private readonly IConfiguration _config;
    private readonly DapperRepository _repo;

    public AuthApiController(IConfiguration config)
    {
        _config = config;
        _repo = new DapperRepository(config);
    }

    [HttpPost("login")]
    public async Task<IActionResult> Login([FromBody] LoginModel model)
    {
        var sql = "SELECT * FROM Users WHERE Username = @Username AND Password = @Password";
        using var conn = _repo.CreateConnection();
        var user = (await conn.QueryFirstOrDefaultAsync<User>(sql, new { model.Username, model.Password }));

        if (user == null)
            return Unauthorized();

        var token = GenerateJwtToken(user);
        return Ok(new { token });
    }

    private string GenerateJwtToken(User user)
    {
        var jwtKey = _config["Jwt:Key"];
        var jwtIssuer = _config["Jwt:Issuer"];

        var claims = new[]
        {
            new Claim(ClaimTypes.Name, user.Username),
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Role, user.Role)
        };

        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var token = new JwtSecurityToken(
            issuer: jwtIssuer,
            audience: null,
            claims: claims,
            expires: DateTime.Now.AddHours(2),
            signingCredentials: creds);

        var handler = new JwtSecurityTokenHandler();
        return handler.WriteToken(token);
    }

    private void SetAuthorizationHeader(HttpClient client)
    {
        var token = HttpContext.Request.Cookies["jwt"];
        if (!string.IsNullOrEmpty(token))
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
    }
}

