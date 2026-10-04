using Microsoft.AspNetCore.Mvc;

namespace CachingServer.Controllers;

[ApiExplorerSettings(IgnoreApi = true)]
[Route("/")]
public class HomeController : Controller
{
    [HttpGet]
    public IActionResult Index()
    {
        return View("Index");
    }
}
