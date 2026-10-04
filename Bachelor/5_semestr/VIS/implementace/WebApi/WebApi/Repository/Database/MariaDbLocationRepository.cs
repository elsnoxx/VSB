using Dapper;
using WebApi.Mappers;
using WebApi.Models.DB;
using WebApi.Models.Domain;
using WebApi.Repository.Database.Implementation;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Repository.Database;

public sealed class MariaDbLocationRepository : ILocationRepository
{
    private readonly IUnitOfWork _uow;
    public MariaDbLocationRepository(IUnitOfWork uow) => _uow = uow;

    public async Task<IEnumerable<Location>> GetAllAsync(CancellationToken ct = default)
    {
        const string sql = """
            SELECT Id, Name, ParentId, CreatedAtUtc
            FROM Locations
            ORDER BY Name
        """;

        var cmd = new CommandDefinition(sql, transaction: _uow.Transaction, cancellationToken: ct);
        var rows = await _uow.Connection.QueryAsync<LocationRow>(cmd);

        return rows.Select(LocationMapper.ToDomain).ToList();
    }

    public async Task<Location?> GetByIdAsync(Guid id, CancellationToken ct = default)
    {
        const string sql = """
            SELECT Id, Name, ParentId, CreatedAtUtc
            FROM Locations
            WHERE Id = @Id
        """;

        var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
        var row = await _uow.Connection.QuerySingleOrDefaultAsync<LocationRow>(cmd);

        return row is null ? null : LocationMapper.ToDomain(row);
    }

    public async Task<bool> ExistsNameAsync(string name, CancellationToken ct = default)
    {
        const string sql = """
            SELECT 1
            FROM Locations
            WHERE Name = @Name
            LIMIT 1
        """;

        var cmd = new CommandDefinition(sql, new { Name = name }, transaction: _uow.Transaction, cancellationToken: ct);
        return (await _uow.Connection.QuerySingleOrDefaultAsync<int?>(cmd)).HasValue;
    }

    public async Task InsertAsync(Location location, CancellationToken ct = default)
    {
        const string sql = """
            INSERT INTO Locations (Id, Name, ParentId, CreatedAtUtc)
            VALUES (@Id, @Name, @ParentId, @CreatedAtUtc)
        """;

        var row = LocationMapper.ToRow(location);
        var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }

    public async Task UpdateAsync(Location location, CancellationToken ct = default)
    {
        const string sql = """
            UPDATE Locations
            SET Name = @Name,
                ParentId = @ParentId
            WHERE Id = @Id
        """;

        var row = LocationMapper.ToRow(location);
        var cmd = new CommandDefinition(sql, row, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }

    public async Task DeleteAsync(Guid id, CancellationToken ct = default)
    {
        const string sql = "DELETE FROM Locations WHERE Id = @Id";
        var cmd = new CommandDefinition(sql, new { Id = id }, transaction: _uow.Transaction, cancellationToken: ct);
        await _uow.Connection.ExecuteAsync(cmd);
    }
}
