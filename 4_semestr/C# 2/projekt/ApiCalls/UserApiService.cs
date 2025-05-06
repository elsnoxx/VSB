using System.Collections.Generic;
using System.Threading.Tasks;
using ApiCalls.Models;

namespace ApiCalls.Services
{
    public class UserApiService
    {
        private readonly ApiClient _apiClient;
        private const string BaseEndpoint = "api/UserApi";

        public UserApiService(ApiClient apiClient)
        {
            _apiClient = apiClient;
        }

        public async Task<List<User>> GetAllUsersAsync()
        {
            return await _apiClient.GetAsync<List<User>>(BaseEndpoint);
        }

        public async Task<User> GetUserByIdAsync(int id)
        {
            return await _apiClient.GetAsync<User>($"{BaseEndpoint}/{id}");
        }

        public async Task CreateUserAsync(User user)
        {
            await _apiClient.PostAsync(BaseEndpoint, user);
        }

        public async Task UpdateUserAsync(int id, User user)
        {
            await _apiClient.PutAsync($"{BaseEndpoint}/{id}", user);
        }

        public async Task DeleteUserAsync(int id)
        {
            await _apiClient.DeleteAsync($"{BaseEndpoint}/{id}");
        }

        public async Task ChangeRoleAsync(int id, string role)
        {
            await _apiClient.PutAsync($"{BaseEndpoint}/{id}/role", new { Role = role });
        }

        public async Task ResetPasswordAsync(int id, string password)
        {
            await _apiClient.PutAsync($"{BaseEndpoint}/{id}/password", new { Password = password });
        }
    }
}