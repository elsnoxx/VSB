using Dapper;
using ParkingLotWEB.Models;
using System.Data;
using Bcrypt.Net;

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
            return await conn.QueryAsync<User>("SELECT Id, Username, Role FROM `User`");
        }

        public async Task<User?> GetByIdAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryFirstOrDefaultAsync<User>("SELECT Id, Username, Role FROM `User` WHERE Id = @id", new { id });
        }

        public async Task<int> UpdateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE `User` SET Username = @Username, Role = @Role WHERE Id = @Id";
            return await conn.ExecuteAsync(sql, new { user.Username, user.Role, user.Id });
        }

        public async Task<int> UpdateAsync(int id, User user)
        {
            using var conn = _dapper.CreateConnection();
            string sql;
            if (!string.IsNullOrEmpty(user.Password))
            {
                sql = "UPDATE `User` SET Username = @Username, Password = @Password, Role = @Role WHERE Id = @Id";
                return await conn.ExecuteAsync(sql, new { user.Username, user.Password, user.Role, Id = id });
            }
            else
            {
                sql = "UPDATE `User` SET Username = @Username, Role = @Role WHERE Id = @Id";
                return await conn.ExecuteAsync(sql, new { user.Username, user.Role, Id = id });
            }
        }

        public async Task<int> CreateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            user.Password = BCrypt.Net.BCrypt.HashPassword(user.Password);
            var sql = "INSERT INTO `User` (Username, Password, Role) VALUES (@Username, @Password, @Role)";
            return await conn.ExecuteAsync(sql, new { user.Username, user.Password, user.Role });
        }

        public async Task<int> DeleteAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "DELETE FROM `User` WHERE Id = @id";
            return await conn.ExecuteAsync(sql, new { id });
        }

        public async Task<User?> AuthenticateAsync(string username, string password)
        {
            var user = await GetByUsernameAsync(username);
            if (user == null)
                return null;
            System.Console.WriteLine($"User found: {user.Username}");
            System.Console.WriteLine($"Password: {password}");
            System.Console.WriteLine($"Hashed Password: {user.Password}");
            System.Console.WriteLine($"Hashed Password: {Bcrypt.Net.BCrypt.HashPassword(password)}");

            if (BCrypt.Net.BCrypt.Verify(password, user.Password))
                return user;

            return null;
        }

        public async Task<int> ChangeRoleAsync(int id, string role)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE `User` SET Role = @Role WHERE Id = @Id";
            return await conn.ExecuteAsync(sql, new { Role = role, Id = id });
        }

        public async Task<int> ResetPasswordAsync(int id, string password)
        {
            using var conn = _dapper.CreateConnection();
            password = BCrypt.Net.BCrypt.HashPassword(password);
            var sql = "UPDATE `User` SET Password = @Password WHERE Id = @Id";
            return await conn.ExecuteAsync(sql, new { Password = password, Id = id });
        }

        public async Task<User?> GetByUsernameAsync(string username)
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryFirstOrDefaultAsync<User>(
                "SELECT * FROM `User` WHERE Username = @username",
                new { username });
        }
    }
}