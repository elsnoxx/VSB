using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.IdentityModel.Tokens;
using System.Net.Http.Headers;
using System.Text;
using System.Security.Claims;

namespace ParkingLotWEB
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // JWT konfigurace
            var jwtKey = builder.Configuration["Jwt:Key"] ?? "tajny_klic";
            var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? "ParkingLotWEB";

            builder.Services.AddAuthentication(options =>
            {
                options.DefaultScheme = CookieAuthenticationDefaults.AuthenticationScheme;
                options.DefaultChallengeScheme = CookieAuthenticationDefaults.AuthenticationScheme;
            })
            .AddCookie(options =>
            {
                options.LoginPath = "/Auth/Login";
                options.AccessDeniedPath = "/Auth/AccessDenied";
            })
            .AddJwtBearer(options =>
            {
                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuer = true,
                    ValidIssuer = builder.Configuration["Jwt:Issuer"],
                    ValidateAudience = false,
                    ValidateLifetime = true,
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"])),
                    ValidateIssuerSigningKey = true,
                    RoleClaimType = ClaimTypes.Role
                };
            });

            builder.Services.ConfigureApplicationCookie(options =>
            {
                options.AccessDeniedPath = "/Auth/AccessDenied";
            });

            builder.Services.AddControllersWithViews();
            builder.Services.AddControllers().AddJsonOptions(options =>
            {
                options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
            });
            builder.Services.AddHttpContextAccessor();
            builder.Services.AddScoped<ParkingLotWEB.Database.UserRepository>();
            builder.Services.AddScoped<ParkingLotWEB.Database.ParkingLotRepository>();
            builder.Services.AddScoped<ParkingLotWEB.Database.ParkingSpaceRepository>();
            builder.Services.AddScoped<ParkingLotWEB.Database.CarRepository>();

            builder.Services.AddHttpClient<ApiClient>(client =>
            {
                // client.BaseAddress = new Uri("https://localhost:7292/");
                client.BaseAddress = new Uri("http://localhost:5062/");
            });
            builder.Services.AddHttpClient();

            var app = builder.Build();

            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Home/Error");
                app.UseHsts();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            app.UseAuthentication();
            app.UseAuthorization();

            // app.UseWhen(context => context.Request.Path.StartsWithSegments("/api"), appBuilder =>
            // {
            //     appBuilder.UseMiddleware<ApiKeyMiddleware>();
            // });

            app.MapControllers();

            app.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");

            app.Run();
        }

        public static void ConfigureHttpClient(HttpClient client, string token)
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
    }
}
