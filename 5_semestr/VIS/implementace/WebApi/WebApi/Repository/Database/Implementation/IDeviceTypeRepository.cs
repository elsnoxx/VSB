using WebApi.Models.Domain;

namespace WebApi.Repository.Database.Implementation
{
    public interface IDeviceTypeRepository
    {
        Task<IEnumerable<DeviceType>> GetAllAsync(CancellationToken ct = default);
        Task<DeviceType?> GetByIdAsync(Guid id, CancellationToken ct = default);
        Task<bool> ExistsNameAsync(string name, CancellationToken ct = default);

        Task InsertAsync(DeviceType type, CancellationToken ct = default);
        Task UpdateAsync(DeviceType type, CancellationToken ct = default);
        Task DeleteAsync(Guid id, CancellationToken ct = default);
    }
}
