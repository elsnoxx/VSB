
using WebApi.DB;
using WebApi.Repository.Database;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.InMemoryDB;
using WebApi.Repository.Unitofwork;
using WebApi.Repository.Unitofwork.Implementation;
using WebApi.Services;
using WebApi.Services.UWO;

namespace WebApi
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            // Services (Use-case layer)
            builder.Services.AddScoped<DeviceService>();
            builder.Services.AddScoped<LocationService>();
            builder.Services.AddScoped<DeviceTypeService>();


            var storage = builder.Configuration["Storage"] ?? "MariaDb";


            if (storage == "MariaDb")
            {
                var connectionString = builder.Configuration.GetConnectionString("MariaDb")
                    ?? throw new InvalidOperationException("Missing connection string 'MariaDb'");

                builder.Services.AddScoped<IDbConnectionFactory>(_ => new MariaDbConnectionFactory(connectionString));

                builder.Services.AddScoped<IDeviceRepository, MariaDbDeviceRepository>();
                builder.Services.AddScoped<ILocationRepository, MariaDbLocationRepository>();
                builder.Services.AddScoped<IDeviceTypeRepository, MariaDbDeviceTypeRepository>();


                builder.Services.AddScoped<IUnitOfWork, MariaDbUnitOfWork>();
            }
            else if (storage == "InMemory")
            {
                // Identity Map musí být sdílená (Singleton), jinak to není Identity Map
                builder.Services.AddSingleton<InMemoryDbContext>(); // nebo InMemoryDbContext

                builder.Services.AddSingleton<IDeviceRepository, InMemoryDeviceRepository>();
                builder.Services.AddSingleton<ILocationRepository, InMemoryLocationRepository>();
                builder.Services.AddSingleton<IDeviceTypeRepository, InMemoryDeviceTypeRepository>();


                builder.Services.AddSingleton<IUnitOfWork, NoOpUnitOfWork>();

            }
            else
            {
                throw new InvalidOperationException($"Unknown Storage value: '{storage}'. Use 'MariaDb' or 'InMemory'.");
            }



            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseHttpsRedirection();

            app.UseAuthorization();


            app.MapControllers();

            app.Run();
        }
    }
}
