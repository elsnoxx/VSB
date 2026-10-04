using Dapper;
using ParkingLotWEB.Models;
using System.Data;
using BCrypt.Net;
using ParkingLotWEB.Exceptions;


namespace ParkingLotWEB.Database
{
    public class UserRepository
    {
        private readonly DapperRepository _dapper;

        public UserRepository(IConfiguration config)
        {
            _dapper = new DapperRepository(config);
        }

        public async Task<IEnumerable<UserDto>> GetAllAsync()
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryAsync<UserDto>("SELECT id, username, password, role, first_name AS FirstName, last_name AS LastName, email FROM `User`");
        }

        public async Task<UserDto?> GetByIdAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryFirstOrDefaultAsync<UserDto>(
                "SELECT Id, Username, Role, password, First_Name AS FirstName, Last_Name AS LastName, Email FROM `User` WHERE Id = @id",
                new { id });
        }

        public async Task<int> UpdateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"UPDATE `User` 
                        SET 
                            Username = @Username, 
                            Password = @Password,
                            Role = @Role, 
                            First_Name = @FirstName, 
                            Last_Name = @LastName, 
                            Email = @Email
                        WHERE Id = @Id";
            return await conn.ExecuteAsync(sql, new 
            { 
                user.Username, 
                user.Password, 
                user.Role, 
                user.FirstName, 
                user.LastName, 
                user.Email, 
                user.Id 
            });
        }

        public async Task<int> CreateAsync(User user)
        {
            using var conn = _dapper.CreateConnection();
            user.Password = BCrypt.Net.BCrypt.HashPassword(user.Password);
            var sql = "INSERT INTO `User` (Username, Password, Role, First_Name, Last_Name, Email) " +
                    "VALUES (@Username, @Password, @Role, @FirstName, @LastName, @Email)";

            try
            {
                return await conn.ExecuteAsync(sql, new
                {
                    user.Username,
                    user.Password,
                    user.Role,
                    user.FirstName,
                    user.LastName,
                    user.Email
                });
            }
            catch (MySql.Data.MySqlClient.MySqlException ex)
            {
                if (ex.Number == 1062)
                {
                    throw new DuplicateUsernameException("Tento uživatel již existuje.", ex);
                }
                throw;
            }
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
        public async Task<List<Car>> GetCarsByUserIdAsync(int userId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"SELECT 
                    car_id AS CarId,
                    user_id AS UserId,
                    license_plate AS LicensePlate,
                    brand_model AS BrandModel,
                    color AS Color
                FROM Car
                WHERE user_id = @userId";
            return (await conn.QueryAsync<Car>(sql, new { userId })).ToList();
        }

        public async Task<List<ParkingHistory>> GetParkingHistoryByUserIdAsync(int userId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"SELECT 
                    c.license_plate AS LicensePlate,
                    p.name AS ParkingLotName,
                    h.arrival_time AS ArrivalTime,
                    h.departure_time AS DepartureTime
                FROM ParkingHistory h
                JOIN Car c ON h.car_id = c.car_id
                JOIN ParkingLot p ON h.parking_lot_id = p.parking_lot_id
                WHERE c.user_id = @userId";
            return (await conn.QueryAsync<ParkingHistory>(sql, new { userId })).ToList();
        }

        public async Task<List<CurrentParking>> GetCurrentParkingByUserIdAsync(int userId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"SELECT 
                    p.parking_lot_id AS ParkingLotId,
                    c.license_plate AS LicensePlate,
                    p.name AS ParkingLotName,
                    o.start_time AS ArrivalTime,
                    ps.parking_space_id AS ParkingSpaceId
                FROM Occupancy o
                JOIN ParkingSpace ps ON o.parking_space_id = ps.parking_space_id
                JOIN ParkingLot p ON ps.parking_lot_id = p.parking_lot_id
                JOIN Car c ON c.license_plate = o.license_plate
                WHERE c.user_id = @userId AND o.end_time IS NULL";
            return (await conn.QueryAsync<CurrentParking>(sql, new { userId })).ToList();
        }

        public async Task<List<ParkingSpace>> GetParkingSpacesWithOwnerAsync(int parkingLotId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"SELECT 
                            ps.parking_space_id AS ParkingSpaceId,
                            ps.space_number AS SpaceNumber,
                            ps.status AS Status,
                            c.user_id AS OwnerId
                        FROM ParkingSpace ps
                        LEFT JOIN Occupancy o ON ps.parking_space_id = o.parking_space_id AND o.end_time IS NULL
                        LEFT JOIN Car c ON o.license_plate = c.license_plate
                        WHERE ps.parking_lot_id = @parkingLotId";
            return (await conn.QueryAsync<ParkingSpace>(sql, new { parkingLotId })).ToList();
        }
    }
}