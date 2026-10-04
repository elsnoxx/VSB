using System.Data;
using MySqlConnector;

namespace WebApi.DB
{
    public interface IDbConnectionFactory
    {
        IDbConnection Create();
    }

    public sealed class MariaDbConnectionFactory : IDbConnectionFactory
    {
        private readonly string _connectionString;

        public MariaDbConnectionFactory(string connectionString)
            => _connectionString = connectionString;

        public IDbConnection Create()
            => new MySqlConnection(_connectionString);
    }
}
