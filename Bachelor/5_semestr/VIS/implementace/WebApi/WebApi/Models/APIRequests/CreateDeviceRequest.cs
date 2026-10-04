namespace WebApi.Models.APIRequests
{
    public class CreateDeviceRequest
    {
        public string SerialNumber { get; set; } = null!;
        public Guid DeviceTypeId { get; set; }
        public string Status { get; set; } = "New";
        public Guid? CurrentLocationId { get; set; }
    }
}
