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

        public async Task<bool> ResetPasswordAsync(int userId, string newPassword)
        {
            var json = JsonSerializer.Serialize(new { Password = newPassword });
            var response = await _httpClientWrapper.PutAsync($"api/UserApi/{userId}/password", json);
            return string.IsNullOrEmpty(response);
        }

        public async Task<bool> CreateUserAsync(User user)
        {
            var options = new JsonSerializerOptions { PropertyNamingPolicy = null };
            var json = JsonSerializer.Serialize(user, options);
            var response = await _httpClientWrapper.PostAsync("api/UserApi", json);
            return string.IsNullOrEmpty(response);
        }

        public async Task<UserProfileViewModel> GetUserProfileAsync(int userId)
        {
            var response = await _httpClientWrapper.GetAsync($"api/UserApi/{userId}/profile");
            return JsonSerializer.Deserialize<UserProfileViewModel>(response);
        }

        public async Task<bool> CreateCarAsync(CarDto car)
        {
            var json = JsonSerializer.Serialize(car);
            var response = await _httpClientWrapper.PostAsync("api/CarApi/new", json);
            return string.IsNullOrEmpty(response);
        }

        public async Task<bool> DeleteCarAsync(int carId)
        {
            var response = await _httpClientWrapper.DeleteAsync($"api/CarApi/{carId}");
            return string.IsNullOrEmpty(response);
        }

    }
}
