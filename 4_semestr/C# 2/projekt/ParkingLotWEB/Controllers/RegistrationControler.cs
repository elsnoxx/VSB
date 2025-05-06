using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using ParkingLotWEB.Database;
using System.Threading.Tasks;
using BCrypt.Net;

public class RegistrationController : Controller
{
    private readonly UserRepository _userRepo;
    public RegistrationController(UserRepository userRepo)
    {
        _userRepo = userRepo;
    }

    [HttpGet]
    public IActionResult Register()
    {
        return View();
    }

    [HttpPost]
    public async Task<IActionResult> Register(RegisterViewModel model)
    {
        if (!ModelState.IsValid)
            return View(model);

        var hashedPassword = BCrypt.Net.BCrypt.HashPassword(model.Password);

        var user = new User
        {
            Username = model.Username,
            Password = hashedPassword,
            Role = model.Role
        };

        await _userRepo.CreateAsync(user);
        return RedirectToAction("Login", "Auth");
    }
}