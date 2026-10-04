using WebApi.DB;
using WebApi.Models.Domain;

namespace WebApi.Models.ActiveRecords
{
    public class LocationRecord
    {
        private readonly InMemoryDbContext _ctx;
        public Location Entity { get; }

        public LocationRecord(InMemoryDbContext ctx, Location entity)
        {
            _ctx = ctx;
            Entity = entity;
        }

        public void Save() => _ctx.Upsert(Entity);
        public void Delete() => _ctx.RemoveLocation(Entity.Id);
    }
}
