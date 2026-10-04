using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using mvcapplikace.Models;
using mvcapplikace.Services;

namespace mvcapplikace.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly MenuService _menuService;

    public HomeController(ILogger<HomeController> logger, MenuService menuService)
    {
        _logger = logger;
        _menuService = menuService;
    }
    public async Task<IActionResult> Index()
    {
        var filteredMenu = await _menuService.GetFilteredMenuAsync();
        return View(filteredMenu);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    


    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
