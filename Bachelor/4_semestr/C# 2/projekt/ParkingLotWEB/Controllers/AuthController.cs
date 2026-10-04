using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Models;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;

public class AuthController : Controller
{
    [HttpGet]
    public IActionResult Login()
    {
        return View();
    }

    [HttpGet]
    public async Task<IActionResult> Logout()
    {
        await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
        // Smažte i JWT cookie, pokud ji používáte
        Response.Cookies.Delete("jwt");
        return RedirectToAction("Login");
    }

    [HttpGet]
    public IActionResult AccessDenied(string returnUrl)
    {
        ViewData["ReturnUrl"] = returnUrl;
        return View();
    }
}