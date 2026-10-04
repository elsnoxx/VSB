using Microsoft.AspNetCore.Http;

namespace cviko2
{
    public class browserChecker
    {
        private readonly RequestDelegate next;
        public browserChecker(RequestDelegate next)
        {
            this.next = next;
        }
        public async Task Invoke(HttpContext ctx, IMyLogger logger)
        {
            bool browser = ctx.Request.Headers.UserAgent.First().Contains("Chrome");
            Console.WriteLine(browser);

            if (browser)
            {
                logger.Log("nice got here");
                //await next(ctx);
                await ctx.Response.WriteAsync("hello");
            }
            else
            {
                ctx.Response.StatusCode = 500;
                await ctx.Response.WriteAsync("Forbiden!");
            }
        }
    }
}
