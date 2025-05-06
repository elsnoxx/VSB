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

        var user = new User
        {
            Username = model.Username,
            Password = model.Password, // zde nehashuj!
            Role = "User",
            FirstName = model.FirstName,
            LastName = model.LastName,
            Email = model.Email
        };

        await _userRepo.CreateAsync(user); // hashování proběhne až v repository
        return RedirectToAction("Login", "Auth");
    }
}