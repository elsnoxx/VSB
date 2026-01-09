namespace WebApi.Models.Domain
{
    public class Device
    {
        public Guid Id { get; }
        public string SerialNumber { get; private set; }
        public Guid DeviceTypeId { get; private set; }
        public string Status { get; private set; }
        public Guid? CurrentLocationId { get; private set; }
        public DateTime CreatedAtUtc { get; }
        public DeviceType DeviceType { get; set; } = null!;

        public Device(Guid id, string serialNumber, Guid deviceTypeId, string status, Guid? currentLocationId, DateTime createdAtUtc)
        {
            Id = id;
            SerialNumber = serialNumber;
            DeviceTypeId = deviceTypeId;
            Status = status;
            CurrentLocationId = currentLocationId;
            CreatedAtUtc = createdAtUtc;
        }

        public void AssignLocation(Guid? locationId) => CurrentLocationId = locationId;
        public void ChangeStatus(string status) => Status = status;
    }
}
