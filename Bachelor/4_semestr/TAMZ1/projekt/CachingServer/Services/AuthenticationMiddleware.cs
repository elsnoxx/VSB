using Microsoft.AspNetCore.Http;
using System.Threading.Tasks;

namespace CachingServer
{
    public class AuthenticationMiddleware
    {
        private readonly RequestDelegate _next;

        public AuthenticationMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task Invoke(HttpContext context)
        {
            // Logování příchozího požadavku
            Console.WriteLine($"=== Příchozí požadavek -> {DateTime.Now}===");
            Console.WriteLine($"Metoda: {context.Request.Method}");
            Console.WriteLine($"Cesta: {context.Request.Path}");
            Console.WriteLine($"Query string: {context.Request.QueryString}");
            Console.WriteLine("Hlavičky:");
            foreach (var header in context.Request.Headers)
            {
                Console.WriteLine($"  {header.Key}: {header.Value}");
            }
            Console.WriteLine("==========================");

            // Povolit přístup ke všem URL v CachingController, health, root a statickým souborům bez autentizace
            if (
                context.Request.Path.StartsWithSegments("/api/Caching") ||
                context.Request.Path.StartsWithSegments("/api/health") ||
                context.Request.Path == "/" ||
                context.Request.Path == "/index.html" ||
                context.Request.Path.Value?.StartsWith("/favicon") == true ||
                context.Request.Path.Value?.StartsWith("/static") == true // pokud máš další složky
            )
            {
                await _next(context);
                return;
            }

            // Povolit OPTIONS požadavky (CORS preflight)
            if (context.Request.Method == "OPTIONS")
            {
                await _next(context);
                return;
            }

            // Ověřit autentizaci pro ostatní požadavky
            if (!context.User.Identity.IsAuthenticated)
            {
                context.Response.StatusCode = 401;
                await context.Response.WriteAsync("Not Authenticated");
                return;
            }

            await _next(context);
        }
    }
}
