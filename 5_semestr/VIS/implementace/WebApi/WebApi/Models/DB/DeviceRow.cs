namespace WebApi.Models.DB
{
    public class DeviceRow
    {
        public Guid Id { get; set; }
        public string SerialNumber { get; set; } = null!;
        public Guid DeviceTypeId { get; set; }
        public string Status { get; set; } = null!;
        public Guid? CurrentLocationId { get; set; }
        public DateTime CreatedAtUtc { get; set; }
    }
}
