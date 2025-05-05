using Dapper;
using ParkingLotWEB.Models;
using System.Data;

namespace ParkingLotWEB.Database
{
    public class ParkingLotRepository
    {
        private readonly DapperRepository _dapper;

        public ParkingLotRepository(IConfiguration config)
        {
            _dapper = new DapperRepository(config);
        }

        public async Task<IEnumerable<ParkingLot>> GetAllAsync()
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryAsync<ParkingLot>("SELECT parking_lot_id AS ParkingLotId, name, latitude, longitude FROM ParkingLot");
        }

        public async Task<ParkingLot?> GetByIdAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryFirstOrDefaultAsync<ParkingLot>(
                "SELECT parking_lot_id AS ParkingLotId, name, latitude, longitude FROM ParkingLot WHERE parking_lot_id = @id", new { id });
        }

        public async Task<int> CreateAsync(ParkingLot lot)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "INSERT INTO ParkingLot (name, latitude, longitude) VALUES (@Name, @Latitude, @Longitude)";
            return await conn.ExecuteAsync(sql, lot);
        }

        public async Task<int> UpdateAsync(ParkingLot lot)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE ParkingLot SET name = @Name, latitude = @Latitude, longitude = @Longitude WHERE parking_lot_id = @ParkingLotId";
            return await conn.ExecuteAsync(sql, lot);
        }

        public async Task<int> DeleteAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "DELETE FROM ParkingLot WHERE parking_lot_id = @id";
            return await conn.ExecuteAsync(sql, new { id });
        }
    }
}