namespace WebApi.Models.APIRequests
{

    public sealed class UpdateDeviceTypeRequest
    {
        public string Name { get; set; } = null!;
        public string? Description { get; set; }
    }
}
