using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ApiCalls.Model;
using System.Text.Json;
using System.Net.Http;

namespace ApiCalls
{
    public class ParkinglotManagement
    {
        private readonly HttpClientWrapper _httpClientWrapper;

        public ParkinglotManagement()
        {
            _httpClientWrapper = new HttpClientWrapper();
        }

        public async Task<List<Parkinglot>> GetAllParkinglotsAsync()
        {
            var response = await _httpClientWrapper.GetAsync("api/ParkinglotApi/withFreespaces");
            return JsonSerializer.Deserialize<List<Parkinglot>>(response);
        }

        public async Task<ParkingLotProfileViewModel> GetParkinglotByIdAsync(int id)
        {
            var response = await _httpClientWrapper.GetAsync($"api/ParkinglotApi/{id}");
            return JsonSerializer.Deserialize<ParkingLotProfileViewModel>(response);
        }

        public async Task<bool> CreateParkinglotAsync(Parkinglot lot)
        {
            var json = JsonSerializer.Serialize(lot);
            var response = await _httpClientWrapper.PostAsync("api/ParkinglotApi", json);
            return !string.IsNullOrEmpty(response);
        }

        public async Task<bool> UpdateParkinglotAsync(Parkinglot lot)
        {
            var json = JsonSerializer.Serialize(lot);
            var response = await _httpClientWrapper.PutAsync($"api/ParkinglotApi/{lot.parkingLotId}", json);
            return string.IsNullOrEmpty(response);
        }

        public async Task<bool> DeleteParkinglotAsync(int id)
        {
            var response = await _httpClientWrapper.DeleteAsync($"api/ParkinglotApi/{id}");
            return string.IsNullOrEmpty(response);
        }

        public async Task<bool> UpdatePricePerHourAsync(int parkingLotId, decimal pricePerHour)
        {
            var json = $"{{\"pricePerHour\":{pricePerHour.ToString(System.Globalization.CultureInfo.InvariantCulture)}}}";
            var response = await _httpClientWrapper.PutAsync($"api/ParkinglotApi/{parkingLotId}/price", json);
            return string.IsNullOrEmpty(response);
        }


    }
}
