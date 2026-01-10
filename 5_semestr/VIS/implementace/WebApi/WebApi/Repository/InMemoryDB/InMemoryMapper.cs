using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Repository.InMemoryDB
{
    public static class InMemoryMapper
    {
        public static DeviceRow ToRow(Device d) => new()
        {
            Id = d.Id,
            SerialNumber = d.SerialNumber,
            DeviceTypeId = d.DeviceTypeId,
            Status = d.Status,
            CurrentLocationId = d.CurrentLocationId,
            CreatedAtUtc = d.CreatedAtUtc
        };

        public static LocationRow ToRow(Location l) => new()
        {
            Id = l.Id,
            Name = l.Name,
            ParentId = l.ParentId,
            CreatedAtUtc = l.CreatedAtUtc
        };

        public static DeviceTypeRow ToRow(DeviceType dt) => new()
        {
            Id = dt.Id,
            Name = dt.Name,
            Description = dt.Description,
            CreatedAtUtc = dt.CreatedAtUtc
        };
    }
}
