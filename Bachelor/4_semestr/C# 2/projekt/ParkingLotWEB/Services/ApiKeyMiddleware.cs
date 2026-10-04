using Microsoft.Extensions.Configuration;

public class ApiKeyMiddleware
{
    private readonly RequestDelegate _next;
    private readonly string _apiKey;
    private const string HEADER_NAME = "X-Api-Key";

    public ApiKeyMiddleware(RequestDelegate next, IConfiguration configuration)
    {
        _next = next;
        _apiKey = configuration["API-KEY:Key"];
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // Výjimka pro login a register endpointy (API i MVC)
        var path = context.Request.Path.Value?.ToLower();
        if (path != null && (
                path.Contains("/login") ||
                path.Contains("/register")
            ))
        {
            await _next(context);
            return;
        }

        var hasApiKey = context.Request.Headers.TryGetValue(HEADER_NAME, out var extractedApiKey);
        string apiKeyLog = hasApiKey ? extractedApiKey.ToString() : "(žádný klíč)";

        Console.WriteLine($"[{DateTime.Now}] API request: {context.Request.Method} {context.Request.Path} from {context.Connection.RemoteIpAddress} | X-Api-Key: {apiKeyLog}");

        if (!hasApiKey || extractedApiKey != _apiKey)
        {
            Console.WriteLine($"[{DateTime.Now}] API Key invalid or missing! Zaslaný: {apiKeyLog}");
            context.Response.StatusCode = 401;
            await context.Response.WriteAsync("API Key is missing or invalid.");
            return;
        }

        await _next(context);
    }
}