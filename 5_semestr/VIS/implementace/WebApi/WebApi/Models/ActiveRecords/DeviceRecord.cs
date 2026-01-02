using WebApi.DB;
using WebApi.Models.Domain;

namespace WebApi.Models.ActiveRecords
{
    public class DeviceRecord
    {
        private readonly InMemoryDbContext _ctx;
        public Device Entity { get; }

        public DeviceRecord(InMemoryDbContext ctx, Device entity)
        {
            _ctx = ctx;
            Entity = entity;
        }

        public void Save() => _ctx.Upsert(Entity);
        public void Delete() => _ctx.RemoveDevice(Entity.Id);
    }
}
