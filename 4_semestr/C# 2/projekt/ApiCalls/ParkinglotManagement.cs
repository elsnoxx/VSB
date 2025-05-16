using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ApiCalls.Model;
using System.Text.Json;

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
    }
}
