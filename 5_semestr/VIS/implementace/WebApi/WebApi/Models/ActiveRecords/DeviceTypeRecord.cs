using WebApi.DB;
using WebApi.Models.Domain;

namespace WebApi.Models.ActiveRecords
{
    public sealed class DeviceTypeRecord
    {
        private readonly InMemoryDbContext _ctx;
        public DeviceType Entity { get; }

        public DeviceTypeRecord(InMemoryDbContext ctx, DeviceType entity)
        {
            _ctx = ctx;
            Entity = entity;
        }

        public void Save() => _ctx.Upsert(Entity);
        public void Delete() => _ctx.RemoveDeviceType(Entity.Id);

    }
}
