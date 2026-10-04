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

        // Zkontroluj, jestli už existuje uživatel se stejným emailem
        var existingUserByEmail = (await _userRepo.GetAllAsync())
            .FirstOrDefault(u => u.Email == model.Email);

        if (existingUserByEmail != null)
        {
            // Přidej pouze informativní hlášku, ale pokračuj v registraci
            ModelState.AddModelError("Email", "Tento email již byl použit, zkus se příhlásit.");
        }

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
            ModelState.AddModelError("Username", ex.Message);
            return View(model);
        }
        catch (Exception ex)
        {
            ModelState.AddModelError("", "Nastala chyba při registraci. Kontaktujte administrátora.");
            return View(model);
        }
    }

}