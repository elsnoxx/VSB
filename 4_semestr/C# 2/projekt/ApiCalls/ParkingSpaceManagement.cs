using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using ApiCalls.Model;

namespace ApiCalls
{
    public class ParkingSpaceManagement
    {
        private readonly HttpClientWrapper _httpClientWrapper;

        public ParkingSpaceManagement()
        {
            _httpClientWrapper = new HttpClientWrapper();
        }

        public async Task<List<ParkingSpace>> GetSpacesByLotIdAsync(int parkingLotId)
        {
            var response = await _httpClientWrapper.GetAsync($"api/ParkingSpaceApi/lot/{parkingLotId}");
            return JsonSerializer.Deserialize<List<ParkingSpace>>(response);
        }

        public async Task<ParkingSpace> GetSpaceByIdAsync(int id)
        {
            var response = await _httpClientWrapper.GetAsync($"api/ParkingSpaceApi/{id}");
            Console.WriteLine($"Response: {response}");
            return JsonSerializer.Deserialize<ParkingSpace>(response);
        }

        public async Task<bool> UpdateSpaceStatusAsync(int parkingSpaceId, string status)
        {
            var body = new { Status = status };
            var json = JsonSerializer.Serialize(body);
            var httpResponse = await _httpClientWrapper.PostAsync($"api/ParkingSpaceApi/status/{parkingSpaceId}", json);
            return string.IsNullOrEmpty(httpResponse);
        }

        public async Task<Occupancy> OccupySpaceAsync(int parkingLotId, OccupyRequest request)
        {
            var json = JsonSerializer.Serialize(request);
            var response = await _httpClientWrapper.PostAsync($"api/ParkingSpaceApi/occupy/{parkingLotId}", json);
            if (!string.IsNullOrEmpty(response))
                return JsonSerializer.Deserialize<Occupancy>(response);
            return null;
        }

        public async Task<bool> ReleaseSpaceAsync(ReleaseRequest request)
        {
            var json = JsonSerializer.Serialize(request);
            var response = await _httpClientWrapper.PostAsync("api/ParkingSpaceApi/release", json);
            return string.IsNullOrEmpty(response);
        }
    }
}