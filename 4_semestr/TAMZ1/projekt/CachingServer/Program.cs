using Microsoft.AspNetCore.Authentication;

namespace CachingServer
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddControllers();

            // Configure CORS
            builder.Services.AddCors(options =>
            {
                options.AddPolicy("AllowAll", policy =>
                {
                    policy.WithOrigins(
                        "http://localhost:8100",
                        "https://localhost",
                        "capacitor://localhost",
                        "http://192.168.208.148:7084"
                    )
                    .AllowAnyHeader()
                    .AllowAnyMethod();
                });
            });

            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            // Povolit statické soubory (např. wwwroot/index.html)
            app.UseStaticFiles(); // Statické soubory z wwwroot

            // Use CORS middleware (musí být před ostatními!)
            app.UseCors("AllowAll");

            // Add Authentication Middleware
            app.UseMiddleware<AuthenticationMiddleware>();

            app.UseAuthorization();

            app.MapControllers(); // API

            // Pokud chceš MVC stránky (např. HomeController s View)
            app.MapDefaultControllerRoute(); // MVC (např. HomeController)

            app.Run();
        }
    }
}
