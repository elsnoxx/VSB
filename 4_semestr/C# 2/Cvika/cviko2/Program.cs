namespace cviko2
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            builder.Services.AddScoped<IMyLogger, JsonLogger>();

            var app = builder.Build();

            

            app.UseMiddleware<ErrorMiddleware>();


            //app.MapGet("/", () => "Hello World!");

            app.UseMiddleware<browserChecker>();

            



            app.Run();
        }
    }
}
