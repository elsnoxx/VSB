using WebApi.Models.APIRequests;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Services.UWO
{
    public class DeviceTypeService
    {
        private readonly IDeviceTypeRepository _repo;
        private readonly IUnitOfWork _uow;

        public DeviceTypeService(IDeviceTypeRepository repo, IUnitOfWork uow)
        {
            _repo = repo;
            _uow = uow;
        }

        public async Task<IEnumerable<DeviceType>> GetAllAsync(CancellationToken ct)
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

        public async Task<DeviceType?> GetByIdAsync(Guid id, CancellationToken ct)
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

        public async Task<Guid> CreateAsync(CreateDeviceTypeRequest req, CancellationToken ct)
        {
            return await _uow.ExecuteInTransactionAsync(async () =>
            {
                if (string.IsNullOrWhiteSpace(req.Name))
                    throw new ArgumentException("Name is required.");

                var name = req.Name.Trim();

                if (await _repo.ExistsNameAsync(name, ct))
                    throw new InvalidOperationException("DEVICETYPE_NAME_DUPLICATE");

                var type = new DeviceType(
                    id: Guid.NewGuid(),
                    name: name,
                    description: string.IsNullOrWhiteSpace(req.Description) ? null : req.Description.Trim(),
                    createdAtUtc: DateTime.UtcNow
                );

                await _repo.InsertAsync(type, ct);
                return type.Id;
            }, ct);
        }

        public async Task UpdateAsync(Guid id, UpdateDeviceTypeRequest req, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var existing = await _repo.GetByIdAsync(id, ct);
                if (existing is null)
                    throw new KeyNotFoundException("DEVICETYPE_NOT_FOUND");

                if (string.IsNullOrWhiteSpace(req.Name))
                    throw new ArgumentException("Name is required.");

                var newName = req.Name.Trim();

                // if name changes, check duplicates
                if (!string.Equals(existing.Name, newName, StringComparison.OrdinalIgnoreCase) &&
                    await _repo.ExistsNameAsync(newName, ct))
                {
                    throw new InvalidOperationException("DEVICETYPE_NAME_DUPLICATE");
                }

                // Pokud máš v DeviceType metody (Rename/SetDescription), použij je.
                // Pokud je entity immutable, vytvoř nový objekt a update přes repo (podle tvého modelu).
                // Zde počítám, že máš settery/metody:
                //existing.Rename(newName);
                //existing.SetDescription(string.IsNullOrWhiteSpace(req.Description) ? null : req.Description.Trim());

                await _repo.UpdateAsync(existing, ct);
            }, ct);
        }

        public async Task DeleteAsync(Guid id, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var existing = await _repo.GetByIdAsync(id, ct);
                if (existing is null)
                    throw new KeyNotFoundException("DEVICETYPE_NOT_FOUND");

                await _repo.DeleteAsync(id, ct);
            }, ct);
        }
    }
}
