namespace WebApi.Models.APIRequests
{
    public sealed class DeviceDetailDto
    {
        public Guid Id { get; init; }
        public string SerialNumber { get; init; } = null!;
        public string Status { get; init; } = null!;
        public DeviceTypeDto DeviceType { get; init; } = null!;
        public DateTime CreatedAtUtc { get; init; }
        public Guid? CurrentLocationId { get; init; }
    }

}
