using WebApi.DB;
using WebApi.Models.ActiveRecords;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;

namespace WebApi.Repository.InMemoryDB
{
    public class InMemoryDeviceTypeRepository : IDeviceTypeRepository
    {
        private readonly InMemoryDbContext _ctx;

        public InMemoryDeviceTypeRepository(InMemoryDbContext ctx) => _ctx = ctx;

        public Task<DeviceType?> GetByIdAsync(Guid id, CancellationToken ct = default)
            => Task.FromResult(_ctx.FindDeviceType(id));

        public Task<IEnumerable<DeviceType>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(_ctx.AllDeviceTypes());

        public Task<bool> ExistsNameAsync(string name, CancellationToken ct = default)
            => Task.FromResult(_ctx.DeviceTypeNameExists(name.Trim()));

        public Task InsertAsync(DeviceType deviceType, CancellationToken ct = default)
        {
            new DeviceTypeRecord(_ctx, deviceType).Save();
            return Task.CompletedTask;
        }

        public Task UpdateAsync(DeviceType deviceType, CancellationToken ct = default)
        {
            new DeviceTypeRecord(_ctx, deviceType).Save();
            return Task.CompletedTask;
        }

        public Task DeleteAsync(Guid id, CancellationToken ct = default)
        {
            var existing = _ctx.FindDeviceType(id);
            if (existing is not null)
                new DeviceTypeRecord(_ctx, existing).Delete();

            return Task.CompletedTask;
        }
    }
}
