using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Repository.Database.Implementation
{
    public interface IDeviceRepository
    {
        Task<IEnumerable<Device>> GetAllAsync(CancellationToken ct = default);
        Task<Device?> GetByIdAsync(Guid id, CancellationToken ct = default);
        Task<bool> ExistsSerialAsync(string serialNumber, CancellationToken ct = default);
        Task InsertAsync(Device device, CancellationToken ct = default);
        Task UpdateAsync(Device device, CancellationToken ct = default);
        Task DeleteAsync(Guid id, CancellationToken ct = default);
        Task<bool> IsLocationOccupiedAsync(Guid locationId, Guid? excludeDeviceId = null, CancellationToken ct = default);
    }
}
