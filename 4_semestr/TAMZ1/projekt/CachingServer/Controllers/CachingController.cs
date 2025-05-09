using CachingServer.Models;
using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace CachingServer.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CachingController : ControllerBase
{
    private static readonly List<Cache> Caches;

    // Fix for CS0119: The issue arises because `File` is being interpreted as the `ControllerBase.File` method instead of `System.IO.File`. 
    // To resolve this, fully qualify the `File` reference with `System.IO`.

    static CachingController()
    {
        var jsonFilePath = Path.Combine(Directory.GetCurrentDirectory(), "caches.json");
        if (System.IO.File.Exists(jsonFilePath))
        {
            var jsonData = System.IO.File.ReadAllText(jsonFilePath);
            Caches = JsonSerializer.Deserialize<List<Cache>>(jsonData) ?? new List<Cache>();
        }
        else
        {
            Caches = new List<Cache>();
        }
    }

    // 1. Seznam všech kešek
    [HttpGet]
    public IActionResult GetAllCaches()
    {
        return Ok(Caches); // Vrací seznam kešek jako JSON
    }

    // 2. Detail kešky podle názvu
    [HttpGet("detail/{name}")]
    public IActionResult GetCacheByName(string name)
    {
        var cache = Caches.Find(c => c.Name?.Equals(name, StringComparison.OrdinalIgnoreCase) == true);
        if (cache == null)
        {
            return NotFound(new { message = "Keška nenalezena." });
        }

        return Ok(cache); // Vrací detail kešky jako JSON
    }


    // 3. Kešky v okolí (GET /api/caching/nearby?lat=...&lng=...&radius=...)
    [HttpGet("nearby")]
    public IActionResult GetNearbyCaches(double lat, double lng, double radius = 1.0)
    {
        var nearbyCaches = new List<Cache>();

        foreach (var cache in Caches)
        {
            var distance = GetDistance(lat, lng, cache.Lat, cache.Lng);
            if (distance <= radius)
            {
                nearbyCaches.Add(cache);
            }
        }

        return Ok(nearbyCaches); // Vrací seznam kešek v okolí jako JSON
    }

    // 4. Vrácení obsahu JSON souboru
    [HttpGet("file")]
    public IActionResult GetJsonFile()
    {
        var jsonFilePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "caches.json");
        if (!System.IO.File.Exists(jsonFilePath))
        {
            return NotFound(new { message = "Soubor nenalezen." });
        }

        var jsonData = System.IO.File.ReadAllText(jsonFilePath);
        return Content(jsonData, "application/json"); // Vrací obsah souboru jako JSON
    }

    // 5. Přidání záznamu do logu, že kešku někdo našel
    [HttpPost("found")]
    public IActionResult LogCacheFound([FromBody] FoundCacheModel model)
    {
        var cache = Caches.Find(c => c.Name?.Equals(model.CacheName, StringComparison.OrdinalIgnoreCase) == true);
        if (cache == null)
        {
            return NotFound(new { message = "Keška nenalezena." });
        }

        if (cache.Logs == null)
        {
            cache.Logs = new List<Log>();
        }

        var logEntry = new Log
        {
            Date = DateTime.UtcNow,
            Finder = model.PersonName
        };

        cache.Logs.Add(logEntry);

        return Ok(new { message = "Log byl úspěšně přidán.", cache });
    }

    // 6. Health check endpoint
    [HttpGet("/api/health")]
    public IActionResult HealthCheck()
    {
        return Ok(new
        {
            status = "Healthy",
            timestamp = DateTime.UtcNow
        });
    }


    // Pomocná metoda pro výpočet vzdálenosti mezi dvěma body (Haversine formula)
    private static double GetDistance(double lat1, double lng1, double lat2, double lng2)
    {
        const double R = 6371e3; // Poloměr Země v metrech
        var φ1 = lat1 * Math.PI / 180;
        var φ2 = lat2 * Math.PI / 180;
        var Δφ = (lat2 - lat1) * Math.PI / 180;
        var Δλ = (lng2 - lng1) * Math.PI / 180;

        var a = Math.Sin(Δφ / 2) * Math.Sin(Δφ / 2) +
                Math.Cos(φ1) * Math.Cos(φ2) *
                Math.Sin(Δλ / 2) * Math.Sin(Δλ / 2);
        var c = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));

        return R * c / 1000; // Vzdálenost v kilometrech
    }
}

public class FoundCacheModel
{
    public string CacheName { get; set; }
    public string PersonName { get; set; }
    public DateTime FoundAt { get; set; }
    public string Distance { get; set; }
}
