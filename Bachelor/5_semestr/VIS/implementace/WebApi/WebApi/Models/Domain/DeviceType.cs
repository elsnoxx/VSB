using static Dapper.SqlMapper;

namespace WebApi.Models.Domain
{
    public class DeviceType
    {
        public Guid Id { get; }
        public string Name { get; private set; }
        public string? Description { get; private set; }
        public DateTime CreatedAtUtc { get; }

        public DeviceType(Guid id, string name, string? description, DateTime createdAtUtc)
        {
            Id = id;
            Name = name;
            Description = description;
            CreatedAtUtc = createdAtUtc;
        }
    }
}
