namespace ParkingLotWEB.Models
{
    public class ApiEndpointViewModel
    {
        public string Controller { get; set; }
        public string Action { get; set; }
        public string HttpMethod { get; set; }
        public string Route { get; set; }
        public List<ApiEndpointParameterViewModel> Parameters { get; set; }
        public string ReturnType { get; set; }
    }
    public class ApiEndpointParameterViewModel
    {
        public string Name { get; set; }
        public string Type { get; set; }
    }
}
