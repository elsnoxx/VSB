using Dapper;
using WebApi.Mappers;
using WebApi.Models.DB;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Repository.Database
{
    public sealed class MariaDbDeviceTypeRepository : IDeviceTypeRepository
    {
        private readonly IUnitOfWork _uow;
        public MariaDbDeviceTypeRepository(IUnitOfWork uow) => _uow = uow;

        public async Task<IEnumerable<DeviceType>> GetAllAsync(CancellationToken ct = default)
        {
            const string sql = """
            SELECT Id, Name, Description, CreatedAtUtc
            FROM DeviceTypes
            ORDER BY Name
        """;

            var cmd = new CommandDefinition(sql, transaction: _uow.Transaction, cancellationToken: ct);
            var rows = await _uow.Connection.QueryAsync<DeviceTypeRow>(cmd);
            return rows.Select(DeviceTypeMapper.ToDomain).ToList();
        }

        public async Task<DeviceType?> GetByIdAsync(Guid id, CancellationToken ct = default)
        {
            const string sql = """
            SELECT Id, Name, Description, CreatedAtUtc
            FROM DeviceTypes
            WHERE Id = @Id
        """;

            var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
            var row = await _uow.Connection.QuerySingleOrDefaultAsync<DeviceTypeRow>(cmd);
            return row is null ? null : DeviceTypeMapper.ToDomain(row);
        }

        public async Task<bool> ExistsNameAsync(string name, CancellationToken ct = default)
        {
            const string sql = """
            SELECT 1 FROM DeviceTypes WHERE Name = @Name LIMIT 1
        """;

            var cmd = new CommandDefinition(sql, new { Name = name }, transaction: _uow.Transaction, cancellationToken: ct);
            return (await _uow.Connection.QuerySingleOrDefaultAsync<int?>(cmd)).HasValue;
        }

        public async Task InsertAsync(DeviceType type, CancellationToken ct = default)
        {
            const string sql = """
            INSERT INTO DeviceTypes (Id, Name, Description, CreatedAtUtc)
            VALUES (@Id, @Name, @Description, @CreatedAtUtc)
        """;

            var row = DeviceTypeMapper.ToRow(type);
            var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
            await _uow.Connection.ExecuteAsync(cmd);
        }

        public async Task UpdateAsync(DeviceType type, CancellationToken ct = default)
        {
            const string sql = """
            UPDATE DeviceTypes
            SET Name = @Name,
                Description = @Description
            WHERE Id = @Id
        """;

            var row = DeviceTypeMapper.ToRow(type);
            var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
            await _uow.Connection.ExecuteAsync(cmd);
        }

        public async Task DeleteAsync(Guid id, CancellationToken ct = default)
        {
            const string sql = "DELETE FROM DeviceTypes WHERE Id = @Id";
            var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
            await _uow.Connection.ExecuteAsync(cmd);
        }
    }
}