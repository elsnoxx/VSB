using WebApi.DB;
using WebApi.Models.ActiveRecords;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;

namespace WebApi.Repository.InMemoryDB
{
    public sealed class InMemoryLocationRepository : ILocationRepository
    {
        private readonly InMemoryDbContext _ctx;

        public InMemoryLocationRepository(InMemoryDbContext ctx) => _ctx = ctx;

        public Task<Location?> GetByIdAsync(Guid id, CancellationToken ct = default)
            => Task.FromResult(_ctx.FindLocation(id));

        public Task<IEnumerable<Location>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(_ctx.AllLocations());

        public Task<bool> ExistsNameAsync(string name, CancellationToken ct = default)
            => Task.FromResult(_ctx.LocationNameExists(name.Trim()));

        public Task InsertAsync(Location location, CancellationToken ct = default)
        {
            new LocationRecord(_ctx, location).Save();
            return Task.CompletedTask;
        }

        public Task UpdateAsync(Location location, CancellationToken ct = default)
        {
            new LocationRecord(_ctx, location).Save();
            return Task.CompletedTask;
        }

        public Task DeleteAsync(Guid id, CancellationToken ct = default)
        {
            var existing = _ctx.FindLocation(id);
            if (existing is not null) new LocationRecord(_ctx, existing).Delete();
            return Task.CompletedTask;
        }
    }
}
