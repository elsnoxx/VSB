using Dapper;
using ParkingLotWEB.Models;

namespace ParkingLotWEB.Database
{
    public class CarRepository
    {
        private readonly DapperRepository _dapper;

        public CarRepository(IConfiguration config)
        {
            _dapper = new DapperRepository(config);
        }

        public async Task<int> CreateAsync(Car car)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "INSERT INTO Car (user_id, license_plate, brand_model, color) VALUES (@UserId, @LicensePlate, @BrandModel, @Color)";
            return await conn.ExecuteAsync(sql, car);
        }

        public async Task<List<Car>> GetByUserIdAsync(int userId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "SELECT * FROM Car WHERE user_id = @userId";
            var cars = await conn.QueryAsync<Car>(sql, new { userId });
            return cars.ToList();
        }

        public async Task<int> DeleteAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "DELETE FROM Car WHERE car_id = @id";
            return await conn.ExecuteAsync(sql, new { id });
        }
    }
}