using WebApi.Models.APIRequests;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Services
{
    public sealed class LocationService
    {
        private readonly ILocationRepository _repo;
        private readonly IUnitOfWork _uow;
        private readonly IDeviceRepository _devices;

        public LocationService(ILocationRepository repo, IDeviceRepository devices, IUnitOfWork uow)
        {
            _repo = repo;
            _devices = devices;
            _uow = uow;
        }

        public async Task<IEnumerable<Location>> GetAllAsync(CancellationToken ct)
        {
            await _uow.OpenAsync(ct);
            try
            {
                return await _repo.GetAllAsync(ct);
            }
            finally
            {
                await _uow.DisposeAsync();
            }
        }

        public async Task<Location?> GetByIdAsync(Guid id, CancellationToken ct)
        {
            await _uow.OpenAsync(ct);
            try
            {
                return await _repo.GetByIdAsync(id, ct);
            }
            finally
            {
                await _uow.DisposeAsync();
            }
        }

        public async Task<Guid> CreateAsync(CreateLocationRequest req, CancellationToken ct)
        {
            return await _uow.ExecuteInTransactionAsync(async () =>
            {
                if (string.IsNullOrWhiteSpace(req.Name))
                    throw new ArgumentException("Name is required.");

                var name = req.Name.Trim();

                if (await _repo.ExistsNameAsync(name, ct))
                    throw new InvalidOperationException("LOCATION_NAME_DUPLICATE");

                var location = new Location(
                    id: Guid.NewGuid(),
                    name: name,
                    parentId: req.ParentId,
                    createdAtUtc: DateTime.UtcNow
                );

                await _repo.InsertAsync(location, ct);
                return location.Id;
            }, ct);
        }

        public async Task UpdateAsync(Guid id, UpdateLocationRequest req, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var existing = await _repo.GetByIdAsync(id, ct);
                if (existing is null)
                    throw new KeyNotFoundException("LOCATION_NOT_FOUND");

                if (string.IsNullOrWhiteSpace(req.Name))
                    throw new ArgumentException("Name is required.");

                var newName = req.Name.Trim();

                if (!string.Equals(existing.Name, newName, StringComparison.OrdinalIgnoreCase) &&
                    await _repo.ExistsNameAsync(newName, ct))
                {
                    throw new InvalidOperationException("LOCATION_NAME_DUPLICATE");
                }

                // doménové chování
                existing.Rename(newName);
                existing.ChangeParent(req.ParentId);

                await _repo.UpdateAsync(existing, ct);
            }, ct);
        }

        public async Task DeleteAsync(Guid id, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var existing = await _repo.GetByIdAsync(id, ct);
                if (existing is null)
                    throw new KeyNotFoundException("LOCATION_NOT_FOUND");

                var occupied = await _devices.IsLocationOccupiedAsync(id, excludeDeviceId: null, ct: ct);
                if (occupied)
                    throw new InvalidOperationException("LOCATION_HAS_DEVICE");

                await _repo.DeleteAsync(id, ct);
            }, ct);
        }

    }
}
