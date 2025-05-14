using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using ApiCalls.Model;

namespace ApiCalls
{
    public class UserManagement
    {
        private readonly HttpClientWrapper _httpClientWrapper;

        public UserManagement()
        {
            _httpClientWrapper = new HttpClientWrapper();
        }

        public async Task<List<User>> GetAllUsersAsync()
        {
            var response = await _httpClientWrapper.GetAsync("api/UserApi");
            Console.WriteLine(response);
            return JsonSerializer.Deserialize<List<User>>(response);
        }

        public async Task UpdateUserAsync(User user)
        {
            var json = JsonSerializer.Serialize(user);
            await _httpClientWrapper.PutAsync($"api/UserApi/{user.Id}", json);
        }
    }
}
