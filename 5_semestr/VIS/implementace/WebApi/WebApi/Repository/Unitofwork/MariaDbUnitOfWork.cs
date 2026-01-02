using System.Data;
using WebApi.DB;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Repository.Unitofwork
{
    public sealed class MariaDbUnitOfWork : IUnitOfWork
    {
        private readonly IDbConnectionFactory _factory;

        public IDbConnection Connection { get; private set; } = null!;
        public IDbTransaction? Transaction { get; private set; }

        public MariaDbUnitOfWork(IDbConnectionFactory factory)
        {
            _factory = factory;
        }
        public Task OpenAsync(CancellationToken ct = default)
        {
            Connection = _factory.Create();
            Connection.Open();
            Transaction = null;
            return Task.CompletedTask;
        }
        public Task BeginAsync(CancellationToken ct = default)
        {
            Connection = _factory.Create();
            Connection.Open();
            Transaction = Connection.BeginTransaction();
            return Task.CompletedTask;
        }

        public Task CommitAsync(CancellationToken ct = default)
        {
            Transaction?.Commit();
            return Task.CompletedTask;
        }

        public Task RollbackAsync(CancellationToken ct = default)
        {
            Transaction?.Rollback();
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync()
        {
            Transaction?.Dispose();
            Connection?.Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
