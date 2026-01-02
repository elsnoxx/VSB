using Microsoft.AspNetCore.Mvc;
using WebApi.Models.APIRequests;
using WebApi.Models.DB;
using WebApi.Models.Domain;
using WebApi.Services;

namespace WebApi.Controllers
{
    [ApiController]
    [Route("api/devices")]
    public class DevicesController : ControllerBase
    {
        private readonly DeviceService _service;

        public DevicesController(DeviceService service) => _service = service;

        [HttpGet]
        public async Task<ActionResult<IReadOnlyList<Device>>> GetAll(CancellationToken ct)
            => Ok(await _service.GetAllAsync(ct));

        [HttpGet("{id:guid}")]
        public async Task<ActionResult<Device>> GetById(Guid id, CancellationToken ct)
        {
            var device = await _service.GetByIdAsync(id, ct);
            return device is null ? NotFound() : Ok(device);
        }

        [HttpPost]
        public async Task<ActionResult<Guid>> Create([FromBody] CreateDeviceRequest req, CancellationToken ct)
        {
            try
            {
                var id = await _service.CreateAsync(req, ct);
                return CreatedAtAction(nameof(GetById), new { id }, id);
            }
            catch (InvalidOperationException ex) when (ex.Message == "DEVICE_SERIAL_DUPLICATE")
            {
                return Conflict("Device with the same serial number already exists.");
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
        }

        [HttpPut("{id:guid}")]
        public async Task<IActionResult> Update(Guid id, [FromBody] UpdateDeviceRequest req, CancellationToken ct)
        {
            try
            {
                await _service.UpdateAsync(id, req, ct);
                return NoContent();
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
        }

        [HttpDelete("{id:guid}")]
        public async Task<IActionResult> Delete(Guid id, CancellationToken ct)
        {
            try
            {
                await _service.DeleteAsync(id, ct);
                return NoContent();
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
        }

        [HttpPut("{id:guid}/assign-location")]
        public async Task<IActionResult> AssignLocation(Guid id, [FromBody] AssignDeviceLocationRequest req, CancellationToken ct)
        {
            try
            {
                await _service.AssignLocationAsync(id, req.LocationId, ct);
                return NoContent();
            }
            catch (KeyNotFoundException e) when (e.Message == "DEVICE_NOT_FOUND")
            {
                return NotFound("Device not found.");
            }
            catch (KeyNotFoundException e) when (e.Message == "LOCATION_NOT_FOUND")
            {
                return NotFound("Location not found.");
            }
            catch (InvalidOperationException e) when (e.Message == "LOCATION_OCCUPIED")
            {
                return Conflict("Location is already occupied by another device.");
            }
        }

    }
}
