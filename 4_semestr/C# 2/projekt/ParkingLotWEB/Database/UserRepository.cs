using Dapper;
using ParkingLotWEB.Models;
using System.Data;

namespace ParkingLotWEB.Database
{
    public class UserRepository
    {
        private readonly DapperRepository _dapper;

        public UserRepository(IConfiguration config)
        {
            _dapper = new DapperRepository(config);
        }

        public async Task<IEnumerable<User>> GetAllAsync()
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryAsync<User>("SELECT Id, Username, Role FROM Users");
        }

        public async Task<User?> GetByIdAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryFirstOrDefaultAsync<User>("SELECT Id, Username, Role FROM Users WHERE Id = @id", new { id });
        }

        public async Task<int> UpdateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE Users SET Username = @Username, Role = @Role WHERE Id = @Id";
            return await conn.ExecuteAsync(sql, new { user.Username, user.Role, user.Id });
        }

        public async Task<int> CreateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "INSERT INTO Users (Username, Password, Role) VALUES (@Username, @Password, @Role)";
            return await conn.ExecuteAsync(sql, new { user.Username, user.Password, user.Role });
        }

        public async Task<int> DeleteAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "DELETE FROM Users WHERE Id = @id";
            return await conn.ExecuteAsync(sql, new { id });
        }
    }
}