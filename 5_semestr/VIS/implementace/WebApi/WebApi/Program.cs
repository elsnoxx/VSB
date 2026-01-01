
using WebApi.DB;
using WebApi.Repository.Database;

namespace WebApi
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            var connectionString = builder.Configuration.GetConnectionString("MariaDb")
                                   ?? throw new InvalidOperationException("Missing connection string 'MariaDb'");

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            builder.Services.AddScoped<IDbConnectionFactory>(_ => new MariaDbConnectionFactory(connectionString));

            var storage = builder.Configuration["Storage"] ?? "MariaDb";

            if (storage == "MariaDb")
            {
                builder.Services.AddScoped<IDeviceRepository, MariaDbDeviceRepository>();
                builder.Services.AddScoped<IUnitOfWork, MariaDbUnitOfWork>();
            }
            else if (storage == "InMemory")
            {
                builder.Services.AddSingleton<IDeviceRepository, InMemoryDeviceRepository>();
                builder.Services.AddSingleton<IUnitOfWork, NoOpUnitOfWork>();
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
