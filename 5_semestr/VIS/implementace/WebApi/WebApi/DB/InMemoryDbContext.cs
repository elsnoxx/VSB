using WebApi.Models.ActiveRecords;
using WebApi.Models.DB;
using WebApi.Models.Domain;

namespace WebApi.DB
{
    public class InMemoryDbContext
    {
        private readonly Dictionary<Guid, Device> _devices = new();
        private readonly Dictionary<Guid, Location> _locations = new();
        private readonly Dictionary<Guid, DeviceType> _deviceTypes = new();

        private readonly Dictionary<string, Guid> _deviceBySerial = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, Guid> _locationByName = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, Guid> _deviceTypeByName = new(StringComparer.OrdinalIgnoreCase);

        // ---- Devices
        public Device? FindDevice(Guid id) => _devices.TryGetValue(id, out var d) ? d : null;
        public IEnumerable<Device> AllDevices() => _devices.Values.ToList();

        public bool DeviceSerialExists(string serial) => _deviceBySerial.ContainsKey(serial);

        internal void Upsert(Device d)
        {
            if (_devices.TryGetValue(d.Id, out var old) &&
                !string.Equals(old.SerialNumber, d.SerialNumber, StringComparison.OrdinalIgnoreCase))
                _deviceBySerial.Remove(old.SerialNumber);

            _devices[d.Id] = d;
            _deviceBySerial[d.SerialNumber] = d.Id;
        }

        internal void RemoveDevice(Guid id)
        {
            if (_devices.TryGetValue(id, out var old))
            {
                _deviceBySerial.Remove(old.SerialNumber);
                _devices.Remove(id);
            }
        }

        // ---- Locations
        public Location? FindLocation(Guid id) => _locations.TryGetValue(id, out var l) ? l : null;
        public IEnumerable<Location> AllLocations() => _locations.Values.OrderBy(x => x.Name).ToList();

        public bool LocationNameExists(string name) => _locationByName.ContainsKey(name);

        internal void Upsert(Location l)
        {
            if (_locations.TryGetValue(l.Id, out var old) &&
                !string.Equals(old.Name, l.Name, StringComparison.OrdinalIgnoreCase))
                _locationByName.Remove(old.Name);

            _locations[l.Id] = l;
            _locationByName[l.Name] = l.Id;
        }

        internal void RemoveLocation(Guid id)
        {
            if (_locations.TryGetValue(id, out var old))
            {
                _locationByName.Remove(old.Name);
                _locations.Remove(id);
            }
        }

        // -- device types
        public DeviceType? FindDeviceType(Guid id) => _deviceTypes.TryGetValue(id, out var dt) ? dt : null;

        public IEnumerable<DeviceType> AllDeviceTypes() => _deviceTypes.Values.OrderBy(x => x.Name).ToList();

        public bool DeviceTypeNameExists(string name)
            => _deviceTypeByName.ContainsKey(name);

        internal void Upsert(DeviceType dt)
        {
            if (_deviceTypes.TryGetValue(dt.Id, out var old) &&
                !string.Equals(old.Name, dt.Name, StringComparison.OrdinalIgnoreCase))
                _deviceTypeByName.Remove(old.Name);
            _deviceTypes[dt.Id] = dt;
            _deviceTypeByName[dt.Name] = dt.Id;
        }

        internal void RemoveDeviceType(Guid id)
        {
            if (_deviceTypes.TryGetValue(id, out var old))
            {
                _deviceTypeByName.Remove(old.Name);
                _deviceTypes.Remove(id);
            }

        }
    }
}
