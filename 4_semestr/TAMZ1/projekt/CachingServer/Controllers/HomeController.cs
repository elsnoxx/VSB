using Microsoft.AspNetCore.Mvc;

namespace CachingServer.Controllers;

[ApiExplorerSettings(IgnoreApi = true)]
[Route("/")]
public class HomeController : ControllerBase
{
    [HttpGet]
    public ContentResult Index()
    {
        
        return Index();
    }
}
