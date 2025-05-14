using Microsoft.AspNetCore.Mvc;
using System.Reflection;
using Microsoft.AspNetCore.Mvc.Routing;
using ParkingLotWEB.Models;

public class ApiDescriptionController : Controller
{
    public IActionResult Index()
    {
        var controllers = Assembly.GetExecutingAssembly()
            .GetTypes()
            .Where(t => typeof(ControllerBase).IsAssignableFrom(t) && t.GetCustomAttribute<ApiControllerAttribute>() != null);

        var endpoints = controllers.SelectMany(controller =>
        {
            var routePrefix = controller.GetCustomAttribute<RouteAttribute>()?.Template ?? "";
            var methods = controller.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);

            return methods
                .Where(m => m.GetCustomAttributes().Any(attr => attr is HttpMethodAttribute))
                .Select(m =>
                {
                    var httpAttr = m.GetCustomAttributes().First(attr => attr is HttpMethodAttribute) as HttpMethodAttribute;
                    var route = httpAttr?.Template != null ? $"{routePrefix}/{httpAttr.Template}".Replace("[controller]", controller.Name.Replace("Controller", "")) : routePrefix.Replace("[controller]", controller.Name.Replace("Controller", ""));
                    var parameters = m.GetParameters().Select(p => new ApiEndpointParameterViewModel
                    {
                        Name = p.Name,
                        Type = p.ParameterType.Name
                    }).ToList();

                    return new ApiEndpointViewModel
                    {
                        Controller = controller.Name.Replace("Controller", ""),
                        Action = m.Name,
                        HttpMethod = httpAttr?.HttpMethods.FirstOrDefault(),
                        Route = route,
                        Parameters = parameters,
                        ReturnType = m.ReturnType.Name
                    };
                });
        }).ToList();

        return View(endpoints);
    }
}