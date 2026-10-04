namespace WebApi.Models.APIRequests
{
    public class UpdateDeviceRequest
    {
        public string Status { get; set; } = null!;
        public Guid? CurrentLocationId { get; set; }
    }

}
