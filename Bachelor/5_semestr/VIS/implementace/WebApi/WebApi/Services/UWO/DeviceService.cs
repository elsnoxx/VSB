using WebApi.Models.APIRequests;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Services
{
    public sealed class DeviceService
    {
        private readonly IDeviceRepository _repo;
        private readonly IDeviceTypeRepository _deviceTypeRepo;
        private readonly IUnitOfWork _uow;
        private readonly ILocationRepository _locations;

        public DeviceService(IDeviceRepository repo, ILocationRepository locations, IUnitOfWork uow, IDeviceTypeRepository deviceTypeRepository)
        {
            _repo = repo;
            _locations = locations;
            _uow = uow;
            _deviceTypeRepo = deviceTypeRepository;
        }

        public async Task<IEnumerable<Device>> GetAllAsync(CancellationToken ct)
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

        public async Task<DeviceDetailDto?> GetByIdAsync(Guid id, CancellationToken ct)
        {
            await _uow.OpenAsync(ct);
            try
            {
                var device = await _repo.GetByIdAsync(id, ct);
                if (device is null) return null;

                var type = await _deviceTypeRepo.GetByIdAsync(device.DeviceTypeId, ct);
                if (type is null)
                    throw new InvalidOperationException("DEVICE_TYPE_MISSING");

                return new DeviceDetailDto
                {
                    Id = device.Id,
                    SerialNumber = device.SerialNumber,
                    Status = device.Status,
                    CurrentLocationId = device.CurrentLocationId,
                    CreatedAtUtc = device.CreatedAtUtc,
                    DeviceType = new DeviceTypeDto
                    {
                        Id = type.Id,
                        Name = type.Name,
                        Description = type.Description
                    }
                };
            }
            finally
            {
                await _uow.DisposeAsync();
            }
        }

        public async Task<Guid> CreateAsync(CreateDeviceRequest request, CancellationToken ct)
        {
            return await _uow.ExecuteInTransactionAsync(async () =>
            {
                if (string.IsNullOrWhiteSpace(request.SerialNumber))
                    throw new ArgumentException("SerialNumber is required.");

                var serial = request.SerialNumber.Trim();

                if (await _repo.ExistsSerialAsync(serial, ct))
                    throw new InvalidOperationException("DEVICE_SERIAL_DUPLICATE");

                var device = new Device(
                    id: Guid.NewGuid(),
                    serialNumber: serial,
                    deviceTypeId: request.DeviceTypeId,
                    status: string.IsNullOrWhiteSpace(request.Status) ? "New" : request.Status.Trim(),
                    currentLocationId: request.CurrentLocationId,
                    createdAtUtc: DateTime.UtcNow
                );

                await _repo.InsertAsync(device, ct);
                return device.Id;
            }, ct);
        }

        public async Task UpdateAsync(Guid id, UpdateDeviceRequest request, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var device = await _repo.GetByIdAsync(id, ct);
                if (device is null)
                    throw new KeyNotFoundException("DEVICE_NOT_FOUND");

                // doménové chování (lepší než device.Status = ...)
                device.ChangeStatus(request.Status);
                device.AssignLocation(request.CurrentLocationId);

                await _repo.UpdateAsync(device, ct);
            }, ct);
        }

        public async Task DeleteAsync(Guid id, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var device = await _repo.GetByIdAsync(id, ct);
                if (device is null)
                    throw new KeyNotFoundException("DEVICE_NOT_FOUND");

                await _repo.DeleteAsync(id, ct);
            }, ct);
        }
        public async Task AssignLocationAsync(Guid deviceId, Guid? locationId, CancellationToken ct)
        {
            await _uow.ExecuteInTransactionAsync(async () =>
            {
                var device = await _repo.GetByIdAsync(deviceId, ct);
                if (device is null)
                    throw new KeyNotFoundException("DEVICE_NOT_FOUND");

                if (locationId is not null)
                {
                    var loc = await _locations.GetByIdAsync(locationId.Value, ct);
                    if (loc is null)
                        throw new KeyNotFoundException("LOCATION_NOT_FOUND");

                    var occupied = await _repo.IsLocationOccupiedAsync(locationId.Value, excludeDeviceId: deviceId, ct: ct);
                    if (occupied)
                        throw new InvalidOperationException("LOCATION_OCCUPIED");
                }

                device.AssignLocation(locationId);
                await _repo.UpdateAsync(device, ct);
            }, ct);
        }

    }
}
