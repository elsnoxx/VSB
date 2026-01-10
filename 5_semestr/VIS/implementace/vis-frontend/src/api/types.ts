export type LocationRow = {
  id: string;
  name: string;
  parentId: string | null;
  createdAtUtc: string;
};

export type CreateLocationRequest = {
  name: string;
  parentId?: string | null;
};

export type DeviceRow = {
  id: string;
  serialNumber: string;
  deviceTypeId?: string; // keep for backward compatibility
  deviceType?: DeviceType; // new: full object from API
  status: string;
  currentLocationId: string | null;
  createdAtUtc: string;
};

export type CreateDeviceRequest = {
  serialNumber: string;
  deviceTypeId: string;
  status?: string;
  currentLocationId?: string | null;
};

export type AssignDeviceLocationRequest = {
  locationId: string | null;
};

export type DeviceType = {
  id: string;
  name: string;
  description?: string;
};