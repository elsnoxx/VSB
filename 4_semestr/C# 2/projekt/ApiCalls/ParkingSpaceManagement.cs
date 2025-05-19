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
            return JsonSerializer.Deserialize<ParkingSpace>(response);
        }

        public async Task<bool> UpdateSpaceAsync(ParkingSpace space)
        {
            var json = JsonSerializer.Serialize(space);
            var response = await _httpClientWrapper.PutAsync($"api/ParkingSpaceApi/{space.parkingSpaceId}", json);
            return string.IsNullOrEmpty(response);
        }

        public async Task<bool> CreateSpaceAsync(ParkingSpace space)
        {
            var json = JsonSerializer.Serialize(space);
            var response = await _httpClientWrapper.PostAsync("api/ParkingSpaceApi", json);
            return !string.IsNullOrEmpty(response);
        }

        public async Task<bool> DeleteSpaceAsync(int id)
        {
            var response = await _httpClientWrapper.DeleteAsync($"api/ParkingSpaceApi/{id}");
            return string.IsNullOrEmpty(response);
        }
    }
}