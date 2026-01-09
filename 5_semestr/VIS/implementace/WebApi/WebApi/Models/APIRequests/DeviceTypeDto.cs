namespace WebApi.Models.APIRequests
{
    public sealed class DeviceTypeDto
    {
        public Guid Id { get; init; }
        public string Name { get; init; } = null!;
        public string? Description { get; init; }
    }
}
