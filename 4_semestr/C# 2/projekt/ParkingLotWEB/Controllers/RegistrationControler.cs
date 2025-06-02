using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using ParkingLotWEB.Database;
using System.Threading.Tasks;
using BCrypt.Net;
using ParkingLotWEB.Exceptions;


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
            Password = model.Password, // hashujeme až v repository
            Role = "User",
            FirstName = model.FirstName,
            LastName = model.LastName,
            Email = model.Email
        };

        try
        {
            await _userRepo.CreateAsync(user);
            return RedirectToAction("Login", "Auth");
        }
        catch (DuplicateUsernameException ex)
        {
            ModelState.AddModelError("Tento email již existuje.", ex.Message);
            return View(model);
        }
        catch (Exception ex)
        {
            ModelState.AddModelError("", "Nastala chyba při registraci. Kontaktujte administrátora.");
            return View(model);
        }
    }

}