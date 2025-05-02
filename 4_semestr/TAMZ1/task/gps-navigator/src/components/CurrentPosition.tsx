import React, { useEffect, useState } from 'react';
import { Geolocation } from '@capacitor/geolocation';

type PositionData = {
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
};

const CurrentPosition: React.FC = () => {
  const [position, setPosition] = useState<PositionData | null>(null);

  useEffect(() => {
    const getCurrentPosition = async () => {
      const coordinates = await Geolocation.getCurrentPosition();
      setPosition({
        lat: coordinates.coords.latitude,
        lng: coordinates.coords.longitude,
        accuracy: coordinates.coords.accuracy,
        altitude: coordinates.coords.altitude,
        altitudeAccuracy: coordinates.coords.altitudeAccuracy,
      });
    };
    getCurrentPosition();
  }, []);

  return (
    <div style={{ padding: 16 }}>
      {position
        ? (
          <div>
            <div>Latitude: {position.lat}</div>
            <div>Longitude: {position.lng}</div>
            <div>Accuracy: {position.accuracy ? `${position.accuracy} m` : 'N/A'}</div>
            <div>Altitude: {position.altitude !== null ? `${position.altitude} m` : 'N/A'}</div>
            <div>Altitude accuracy: {position.altitudeAccuracy !== null ? `${position.altitudeAccuracy} m` : 'N/A'}</div>
          </div>
        )
        : <div>Loading position...</div>
      }
    </div>
  );
};

export default CurrentPosition;