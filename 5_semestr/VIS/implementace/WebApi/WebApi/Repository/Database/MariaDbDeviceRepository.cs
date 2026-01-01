using Dapper;
using System.Data;
using WebApi.DB;
using WebApi.Models;

namespace WebApi.Repository.Database
{
    public sealed class MariaDbDeviceRepository
    {
        private readonly IDbConnectionFactory _factory;

        public MariaDbDeviceRepository(IDbConnectionFactory factory)
            => _factory = factory;

        public async Task<DeviceRow?> GetByIdAsync(Guid id, CancellationToken ct = default)
        {
            const string sql = """
            SELECT Id, SerialNumber, DeviceTypeId, Status, CurrentLocationId, CreatedAtUtc
            FROM Devices
            WHERE Id = @Id
        """;

            using var db = _factory.Create();
            var cmd = new CommandDefinition(sql, new { Id = id }, cancellationToken: ct);

            return await db.QuerySingleOrDefaultAsync<DeviceRow>(cmd);
        }

        public async Task<bool> ExistsSerialAsync(string serialNumber, CancellationToken ct = default)
        {
            const string sql = """
            SELECT 1
            FROM Devices
            WHERE SerialNumber = @SerialNumber
            LIMIT 1
        """;

            using var db = _factory.Create();
            var cmd = new CommandDefinition(sql, new { SerialNumber = serialNumber }, cancellationToken: ct);

            var result = await db.QuerySingleOrDefaultAsync<int?>(cmd);
            return result.HasValue;
        }

        public async Task InsertAsync(DeviceRow row, CancellationToken ct = default)
        {
            const string sql = """
            INSERT INTO Devices (Id, SerialNumber, DeviceTypeId, Status, CurrentLocationId, CreatedAtUtc)
            VALUES (@Id, @SerialNumber, @DeviceTypeId, @Status, @CurrentLocationId, @CreatedAtUtc)
        """;

            using var db = _factory.Create();
            var cmd = new CommandDefinition(sql, row, cancellationToken: ct);

            await db.ExecuteAsync(cmd);
        }

        public async Task UpdateAsync(DeviceRow row, CancellationToken ct = default)
        {
            const string sql = """
            UPDATE Devices
            SET SerialNumber = @SerialNumber,
                DeviceTypeId = @DeviceTypeId,
                Status = @Status,
                CurrentLocationId = @CurrentLocationId
            WHERE Id = @Id
        """;

            using var db = _factory.Create();
            var cmd = new CommandDefinition(sql, row, cancellationToken: ct);

            await db.ExecuteAsync(cmd);
        }

        public async Task DeleteAsync(Guid id, CancellationToken ct = default)
        {
            const string sql = """
            DELETE FROM Devices
            WHERE Id = @Id
        """;

            using var db = _factory.Create();
            var cmd = new CommandDefinition(sql, new { Id = id }, cancellationToken: ct);

            await db.ExecuteAsync(cmd);
        }
    }
}
