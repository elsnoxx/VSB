using WebApi.DB;
using WebApi.Models.ActiveRecords;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;

namespace WebApi.Repository.InMemoryDB
{
    public sealed class InMemoryDeviceRepository : IDeviceRepository
    {
        private readonly InMemoryDbContext _ctx;

        public InMemoryDeviceRepository(InMemoryDbContext ctx) => _ctx = ctx;

        public Task<Device?> GetByIdAsync(Guid id, CancellationToken ct = default)
            => Task.FromResult(_ctx.FindDevice(id));

        public Task<IEnumerable<Device>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(_ctx.AllDevices());

        public Task<bool> ExistsSerialAsync(string serialNumber, CancellationToken ct = default)
            => Task.FromResult(_ctx.DeviceSerialExists(serialNumber.Trim()));

        public Task InsertAsync(Device device, CancellationToken ct = default)
        {
            new DeviceRecord(_ctx, device).Save(); // Active Record Save
            return Task.CompletedTask;
        }

        public Task UpdateAsync(Device device, CancellationToken ct = default)
        {
            // Identity Map: chceme držet jednu instanci – update ideálně děláš na té instanci,
            // ale když ti service pošle instanci, která už je v mapě, tak je to OK.
            // Pokud by to byla jiná instance se stejným ID, tak ji přepíšeš.
            new DeviceRecord(_ctx, device).Save();
            return Task.CompletedTask;
        }

        public Task DeleteAsync(Guid id, CancellationToken ct = default)
        {
            var existing = _ctx.FindDevice(id);
            if (existing is not null) new DeviceRecord(_ctx, existing).Delete();
            return Task.CompletedTask;
        }

        public Task<bool> IsLocationOccupiedAsync(Guid locationId, Guid? excludeDeviceId = null, CancellationToken ct = default)
        {
            var occupied = _ctx.AllDevices().Any(d =>
                d.CurrentLocationId == locationId &&
                (excludeDeviceId is null || d.Id != excludeDeviceId));

            return Task.FromResult(occupied);
        }

    }
}
