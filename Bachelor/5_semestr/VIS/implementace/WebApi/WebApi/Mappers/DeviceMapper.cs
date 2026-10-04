using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Mappers
{
    public class DeviceMapper
    {
        public static Device ToDomain(DeviceRow row) =>
        new(row.Id, row.SerialNumber, row.DeviceTypeId, row.Status, row.CurrentLocationId, row.CreatedAtUtc);

        public static DeviceRow ToRow(Device entity) => new()
        {
            Id = entity.Id,
            SerialNumber = entity.SerialNumber,
            DeviceTypeId = entity.DeviceTypeId,
            Status = entity.Status,
            CurrentLocationId = entity.CurrentLocationId,
            CreatedAtUtc = entity.CreatedAtUtc
        };
    }
}
