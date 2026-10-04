namespace WebApi.Models.APIRequests
{
    public sealed class CreateDeviceTypeRequest
    {
        public string Name { get; set; } = null!;
        public string? Description { get; set; }
    }
}
