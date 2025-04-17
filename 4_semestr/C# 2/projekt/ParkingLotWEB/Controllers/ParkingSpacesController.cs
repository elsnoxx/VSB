using Microsoft.AspNetCore.Mvc;
using ParkingLotWEB.Database;

namespace ParkingLotWEB.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ParkingSpacesController : ControllerBase
    {
        private readonly ParkingSpaceRepository _repository;

        public ParkingSpacesController(IConfiguration configuration)
        {
            _repository = new ParkingSpaceRepository(configuration);
        }

        [HttpGet("parkinglot/{parkingLotId}")]
        public async Task<IActionResult> GetByParkingLot(int parkingLotId)
        {
            var spaces = await _repository.GetByParkingLotIdAsync(parkingLotId);
            return Ok(spaces);
        }

        [HttpGet("available/{parkingLotId}")]
        public async Task<IActionResult> GetAvailable(int parkingLotId)
        {
            var availableSpaces = await _repository.GetAvailableSpacesAsync(parkingLotId);
            return Ok(availableSpaces);
        }

        [HttpPut("{id}/status")]
        public async Task<IActionResult> UpdateStatus(int id, [FromBody] string newStatus)
        {
            // Zde by měla být validace stavu před aktualizací
            if (newStatus != "available" && newStatus != "occupied" && newStatus != "under_maintenance")
            {
                return BadRequest("Invalid status value");
            }

            bool result = await _repository.UpdateStatusAsync(id, newStatus);
            if (result)
            {
                return Ok();
            }
            return BadRequest("Could not update status");
        }
    }
}
