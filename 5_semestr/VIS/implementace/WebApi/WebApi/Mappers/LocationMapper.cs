using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Mappers
{
    public class LocationMapper
    {
        public static Location ToDomain(LocationRow row) =>
        new(row.Id, row.Name, row.ParentId, row.CreatedAtUtc);

        public static LocationRow ToRow(Location entity) => new()
        {
            Id = entity.Id,
            Name = entity.Name,
            ParentId = entity.ParentId,
            CreatedAtUtc = entity.CreatedAtUtc
        };
    }
}
