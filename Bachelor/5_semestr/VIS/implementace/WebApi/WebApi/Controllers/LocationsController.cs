using Microsoft.AspNetCore.Mvc;
using WebApi.Models.APIRequests;
using WebApi.Models.DB;
using WebApi.Models.Domain;
using WebApi.Services;

namespace WebApi.Controllers
{
    [ApiController]
    [Route("api/locations")]
    public class LocationsController : ControllerBase
    {
        private readonly LocationService _service;

        public LocationsController(LocationService service) => _service = service;

        [HttpGet]
        public async Task<ActionResult<IReadOnlyList<Location>>> GetAll(CancellationToken ct)
            => Ok(await _service.GetAllAsync(ct));

        [HttpGet("{id:guid}")]
        public async Task<ActionResult<Location>> GetById(Guid id, CancellationToken ct)
        {
            var row = await _service.GetByIdAsync(id, ct);
            return row is null ? NotFound() : Ok(row);
        }

        [HttpPost]
        public async Task<ActionResult<Guid>> Create([FromBody] CreateLocationRequest req, CancellationToken ct)
        {
            try
            {
                var id = await _service.CreateAsync(req, ct);
                return CreatedAtAction(nameof(GetById), new { id }, id);
            }
            catch (InvalidOperationException ex) when (ex.Message == "LOCATION_NAME_DUPLICATE")
            {
                return Conflict("Location with this name already exists.");
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
        }

        [HttpPut("{id:guid}")]
        public async Task<IActionResult> Update(Guid id, [FromBody] UpdateLocationRequest req, CancellationToken ct)
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
            catch (InvalidOperationException ex) when (ex.Message == "LOCATION_NAME_DUPLICATE")
            {
                return Conflict("Location with this name already exists.");
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
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
            catch (InvalidOperationException e) when (e.Message == "LOCATION_HAS_DEVICE")
            {
                return Conflict("Location cannot be deleted because a device is assigned to it.");
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
        }
    }
}
