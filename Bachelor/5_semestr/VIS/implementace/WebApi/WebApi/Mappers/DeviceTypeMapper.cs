using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Mappers
{
    public class DeviceTypeMapper
    {
        public static DeviceType ToDomain(DeviceTypeRow row)
        => new(row.Id, row.Name, row.Description, row.CreatedAtUtc);

        public static DeviceTypeRow ToRow(DeviceType entity)
            => new()
            {
                Id = entity.Id,
                Name = entity.Name,
                Description = entity.Description,
                CreatedAtUtc = entity.CreatedAtUtc
            };
    }
}
