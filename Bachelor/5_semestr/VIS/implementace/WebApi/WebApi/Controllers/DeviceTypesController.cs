using Microsoft.AspNetCore.Mvc;
using WebApi.Models.APIRequests;
using WebApi.Models.Domain;
using WebApi.Services.UWO;

namespace WebApi.Controllers
{
    [ApiController]
    [Route("api/device-types")]
    public sealed class DeviceTypesController : ControllerBase
    {
        private readonly DeviceTypeService _service;

        public DeviceTypesController(DeviceTypeService service) => _service = service;

        [HttpGet]
        public async Task<ActionResult<IReadOnlyList<DeviceType>>> GetAll(CancellationToken ct)
            => Ok(await _service.GetAllAsync(ct));

        [HttpGet("{id:guid}")]
        public async Task<ActionResult<DeviceType>> GetById(Guid id, CancellationToken ct)
        {
            var item = await _service.GetByIdAsync(id, ct);
            return item is null ? NotFound() : Ok(item);
        }

        [HttpPost]
        public async Task<ActionResult<Guid>> Create([FromBody] CreateDeviceTypeRequest req, CancellationToken ct)
        {
            try
            {
                var id = await _service.CreateAsync(req, ct);
                return CreatedAtAction(nameof(GetById), new { id }, id);
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
            catch (InvalidOperationException ex) when (ex.Message == "DEVICETYPE_NAME_DUPLICATE")
            {
                return Conflict("Device type with the same name already exists.");
            }
        }

        [HttpPut("{id:guid}")]
        public async Task<IActionResult> Update(Guid id, [FromBody] UpdateDeviceTypeRequest req, CancellationToken ct)
        {
            try
            {
                await _service.UpdateAsync(id, req, ct);
                return NoContent();
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (InvalidOperationException ex) when (ex.Message == "DEVICETYPE_NAME_DUPLICATE")
            {
                return Conflict("Device type with the same name already exists.");
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
    }
}
