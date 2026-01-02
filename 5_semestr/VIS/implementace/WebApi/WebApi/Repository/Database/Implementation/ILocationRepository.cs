using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.Repository.Database.Implementation
{
    public interface ILocationRepository
    {
        Task<IEnumerable<Location>> GetAllAsync(CancellationToken ct = default);
        Task<Location?> GetByIdAsync(Guid id, CancellationToken ct = default);
        Task<bool> ExistsNameAsync(string name, CancellationToken ct = default);
        Task InsertAsync(Location location, CancellationToken ct = default);
        Task UpdateAsync(Location location, CancellationToken ct = default);
        Task DeleteAsync(Guid id, CancellationToken ct = default);
    }

}
