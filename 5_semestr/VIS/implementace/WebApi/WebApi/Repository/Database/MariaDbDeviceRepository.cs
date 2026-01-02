using Dapper;
using WebApi.Mappers;
using WebApi.Models.ActiveRecords;
using WebApi.Models.DB;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Repository.Database;

public sealed class MariaDbDeviceRepository : IDeviceRepository
{
    private readonly IUnitOfWork _uow;
    public MariaDbDeviceRepository(IUnitOfWork uow) => _uow = uow;

    public async Task<IEnumerable<Device>> GetAllAsync(CancellationToken ct = default)
    {
        const string sql = """
            SELECT Id, SerialNumber, DeviceTypeId, Status, CurrentLocationId, CreatedAtUtc
            FROM Devices
        """;

        var cmd = new CommandDefinition(sql, transaction: _uow.Transaction, cancellationToken: ct);
        var rows = await _uow.Connection.QueryAsync<DeviceRow>(cmd);

        return rows.Select(DeviceMapper.ToDomain).ToList();
    }

    public async Task<Device?> GetByIdAsync(Guid id, CancellationToken ct = default)
    {
        const string sql = """
            SELECT Id, SerialNumber, DeviceTypeId, Status, CurrentLocationId, CreatedAtUtc
            FROM Devices
            WHERE Id = @Id
        """;

        var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
        var row = await _uow.Connection.QuerySingleOrDefaultAsync<DeviceRow>(cmd);

        return row is null ? null : DeviceMapper.ToDomain(row);
    }

    public async Task<bool> ExistsSerialAsync(string serialNumber, CancellationToken ct = default)
    {
        const string sql = """
            SELECT 1
            FROM Devices
            WHERE SerialNumber = @SerialNumber
            LIMIT 1
        """;

        var cmd = new CommandDefinition(sql, new { SerialNumber = serialNumber }, transaction: _uow.Transaction, cancellationToken: ct);
        return (await _uow.Connection.QuerySingleOrDefaultAsync<int?>(cmd)).HasValue;
    }

    public async Task InsertAsync(Device device, CancellationToken ct = default)
    {
        const string sql = """
            INSERT INTO Devices (Id, SerialNumber, DeviceTypeId, Status, CurrentLocationId, CreatedAtUtc)
            VALUES (@Id, @SerialNumber, @DeviceTypeId, @Status, @CurrentLocationId, @CreatedAtUtc)
        """;

        var row = DeviceMapper.ToRow(device);
        var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }

    public async Task UpdateAsync(Device device, CancellationToken ct = default)
    {
        const string sql = """
            UPDATE Devices
            SET SerialNumber = @SerialNumber,
                DeviceTypeId = @DeviceTypeId,
                Status = @Status,
                CurrentLocationId = @CurrentLocationId
            WHERE Id = @Id
        """;

        var row = DeviceMapper.ToRow(device);
        var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }

    public async Task DeleteAsync(Guid id, CancellationToken ct = default)
    {
        const string sql = "DELETE FROM Devices WHERE Id = @Id";
        var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }

    public async Task<bool> IsLocationOccupiedAsync(Guid locationId, Guid? excludeDeviceId = null, CancellationToken ct = default)
    {
        const string sql = """
        SELECT 1
        FROM Devices
        WHERE CurrentLocationId = @LocationId
          AND (@ExcludeId IS NULL OR Id <> @ExcludeId)
        LIMIT 1
    """;

        var cmd = new CommandDefinition(
            sql,
            new { LocationId = locationId, ExcludeId = excludeDeviceId },
            transaction: _uow.Transaction,
            cancellationToken: ct);

        return (await _uow.Connection.QuerySingleOrDefaultAsync<int?>(cmd)).HasValue;
    }

}
