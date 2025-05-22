using Microsoft.AspNetCore.Mvc;
using System.Reflection;
using Microsoft.AspNetCore.Mvc.Routing;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.ViewModels;
using ParkingLotWEB.Models.Entities;
using System.Threading.Tasks;

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

                    // Získat skutečný návratový typ z ProducesResponseTypeAttribute
                    var responseTypeAttr = m.GetCustomAttributes().FirstOrDefault(a => a is ProducesResponseTypeAttribute) as ProducesResponseTypeAttribute;
                    var actualReturnType = responseTypeAttr?.Type ?? GetReturnInnerType(m.ReturnType);

                    return new ApiEndpointViewModel
                    {
                        Controller = controller.Name.Replace("Controller", ""),
                        Action = m.Name,
                        HttpMethod = httpAttr?.HttpMethods.FirstOrDefault(),
                        Route = route,
                        Parameters = parameters,
                        ReturnType = GetFriendlyReturnType(actualReturnType),
                        ReturnJsonExample = GetJsonExample(actualReturnType)
                    };
                });
        }).ToList();

        return View(endpoints);
    }

    private string GetFriendlyReturnType(Type returnType)
    {
        // Pokud je Task<T>, vezmi T
        if (returnType.IsGenericType && returnType.GetGenericTypeDefinition() == typeof(Task<>))
            return GetFriendlyReturnType(returnType.GetGenericArguments()[0]);

        // Pokud je ActionResult<T> nebo podobné, vezmi T
        if (returnType.IsGenericType && (
            returnType.GetGenericTypeDefinition() == typeof(ActionResult<>) ||
            returnType.GetGenericTypeDefinition().Name.StartsWith("ActionResult")))
        {
            var innerType = returnType.GetGenericArguments()[0];
            if (innerType == typeof(void))
                return "HTTP Status Codes: 204 No Content, 404 Not Found nebo 400 Bad Request";
            return innerType.Name + GetJsonExample(innerType);
        }

        // Pokud je IActionResult nebo ActionResult, zobraz status kódy
        if (returnType == typeof(IActionResult) || returnType == typeof(ActionResult))
            return "HTTP Status Codes: 204 No Content, 404 Not Found nebo 400 Bad Request";

        // Pokud je void, zobraz status kód 204
        if (returnType == typeof(void))
            return "204 No Content";

        // Jinak vrať název typu a ukázku JSONu
        return returnType.Name + GetJsonExample(returnType);
    }

    // Vrací ukázku JSONu pro typ
    private string GetJsonExample(Type type)
    {
        if (type == typeof(void) || type == typeof(IActionResult) || type == typeof(ActionResult))
            return "";
            
        // Zpracování Task<T>
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Task<>))
            type = type.GetGenericArguments()[0];
            
        // Zpracování ActionResult<T>
        if (type.IsGenericType && (
            type.GetGenericTypeDefinition() == typeof(ActionResult<>)))
        {
            type = type.GetGenericArguments()[0];
        }
        
        // Zpracování kolekcí (IEnumerable, List, atd.)
        if (type.IsGenericType && (
            type.GetGenericTypeDefinition() == typeof(IEnumerable<>) ||
            type.GetGenericTypeDefinition() == typeof(List<>) ||
            type.GetGenericTypeDefinition() == typeof(ICollection<>) ||
            type.GetGenericTypeDefinition() == typeof(IList<>)))
        {
            var itemType = type.GetGenericArguments()[0];
            var itemExample = GetJsonExampleForType(itemType);
            return $"[ {itemExample} ]";
        }
        
        return GetJsonExampleForType(type);
    }

    private string GetJsonExampleForType(Type type)
    {
        string typeName = type.Name.ToLowerInvariant();
        string typeFullName = type.FullName?.ToLowerInvariant() ?? "";
        
        // Car a jeho varianty
        if (typeName == "car" || typeFullName.Contains(".car") || 
            typeName.Contains("car") || type.Name == "Car" || 
            type.Name.EndsWith("CarDto"))
        {
            return "{ \"carId\": 1, \"userId\": 1, \"licensePlate\": \"1A23456\", \"brandModel\": \"Škoda Octavia\", \"color\": \"červená\" }";
        }
        
        // Další typy
        if (type.Name == "TokenResponse" || typeFullName.Contains("tokenresponse"))
            return "{ \"token\": \"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...\" }";
            
        if (typeName.Contains("parkinglot") && !typeName.Contains("dto"))
            return "{ \"parkingLotId\": 1, \"name\": \"Central Parking\", \"latitude\": 49.8175, \"longitude\": 18.2528, \"capacity\": 100, \"freeSpaces\": 25, \"pricePerHour\": 30 }";
            
        if (typeName.Contains("parkinglotdto") || typeFullName.Contains("parkinglotdto"))
            return "{ \"parkingLotId\": 1, \"name\": \"Central Parking\", \"latitude\": 49.8175, \"longitude\": 18.2528, \"capacity\": 100, \"freeSpaces\": 25, \"pricePerHour\": 30, \"parkingSpaces\": [...] }";
            
        if (typeName == "user" || typeFullName.Contains(".user"))
            return "{ \"id\": 1, \"username\": \"jnovak\", \"firstName\": \"Jan\", \"lastName\": \"Novák\", \"email\": \"jan.novak@example.com\", \"role\": \"User\" }";


        if (typeName == "occupancy" || typeName == "Occupancy" || typeFullName.Contains(".occupancy"))
            return "{ \"occupancyId\": 1, \"parkingSpaceId\": 1, \"licensePlate\": \"1A23456\", \"startTime\": \"2023-10-01T12:00:00\", \"endTime\": \"2023-10-01T14:00:00\" }";

        if (typeName == "statushistory" || typeFullName.Contains(".statushistory") || typeName == "StatusHistory")
            return "{ \"historyId\": 1, \"parkingSpaceId\": 1, \"SpaceNumber\": 1 , \"ParkingLotId\" : 1,\"status\": \"occupied\", \"timestamp\": \"2023-10-01T12:00:00\" }";
        
        if (typeName == "parkinghistory" || typeFullName.Contains(".parkinghistory") || typeName == "ParkingHistory")
            return "{ \"historyId\": 1, \"licensePlate\": \"1A23456\", \"parkingLotName\": \"Central Parking\", \"arrivalTime\": \"2023-10-01T12:00:00\", \"departureTime\": \"2023-10-01T14:00:00\", \"duration\": 120, \"pricePerHour\": 30, \"totalPrice\": 60 }";
        
        if (typeName == "userprofileviewmodel" || typeFullName.Contains(".userprofileviewmodel"))
            return "{ \"id\": 1, \"username\": \"jnovak\", \"firstName\": \"Jan\", \"lastName\": \"Novák\", \"email\": \"jan.novak@example.com\", \"role\": \"User\", \"password\": \"***\", \"cars\": [{ \"carId\": 1, \"userId\": 1, \"licensePlate\": \"1A23456\", \"brandModel\": \"Škoda Octavia\", \"color\": \"červená\" }], \"parkingHistory\": [{ \"historyId\": 1, \"licensePlate\": \"1A23456\", \"parkingLotName\": \"Central Parking\", \"arrivalTime\": \"2023-10-01T12:00:00\", \"departureTime\": \"2023-10-01T14:00:00\", \"duration\": 120, \"pricePerHour\": 30, \"totalPrice\": 60 }], \"currentParking\": [{ \"licensePlate\": \"1A23456\", \"parkingLotName\": \"Central Parking\", \"arrivalTime\": \"2023-10-01T12:00:00\", \"parkingSpaceNumber\": 42 }] }";


        // Obecné zpracování - získání vlastností typu
        var props = type.GetProperties();
        if (props.Length == 0) 
            return "{}";
            
        return "{ " + string.Join(", ", props.Select(p => $"\"{p.Name}\": \"...\"")) + " }";
    }

    private Type GetReturnInnerType(Type returnType)
    {
        if (returnType.IsGenericType && returnType.GetGenericTypeDefinition() == typeof(Task<>))
            return returnType.GetGenericArguments()[0];
        return returnType;
    }
}