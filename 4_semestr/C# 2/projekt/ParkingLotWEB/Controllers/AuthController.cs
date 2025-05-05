using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;

public class AuthController : Controller
{
    [HttpGet]
    public IActionResult Login()
    {
        return View();
    }
}