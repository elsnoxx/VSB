import React, { useEffect, useState, useRef } from 'react';
import { Geolocation } from '@capacitor/geolocation';
import { IonGrid, IonRow, IonCol, IonCard, IonCardContent, IonText, IonSpinner } from '@ionic/react';

type PositionData = {
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
  timestamp?: number;
};

interface CurrentPositionProps {
  onPosition?: (pos: { lat: number; lng: number }) => void;
}

const toRad = (deg: number) => (deg * Math.PI) / 180;

// Výpočet vzdálenosti v metrech mezi dvěma body (Haversine)
function getDistance(lat1: number, lng1: number, lat2: number, lng2: number) {
  const R = 6371000; // poloměr Země v metrech
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLng / 2) * Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Výpočet azimutu (směru) v stupních
function getBearing(lat1: number, lng1: number, lat2: number, lng2: number) {
  const dLng = toRad(lng2 - lng1);
  const y = Math.sin(dLng) * Math.cos(toRad(lat2));
  const x =
    Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
    Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLng);
  const brng = Math.atan2(y, x);
  return ((brng * 180) / Math.PI + 360) % 360;
}

const CurrentPosition: React.FC<CurrentPositionProps> = ({ onPosition }) => {
  const [position, setPosition] = useState<PositionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState<number | null>(null); // m/s
  const [bearing, setBearing] = useState<number | null>(null); // deg

  // Uložíme předchozí pozici a čas
  const prevPos = useRef<PositionData | null>(null);

  useEffect(() => {
    let watchId: string | null = null;

    Geolocation.watchPosition(
      { enableHighAccuracy: true },
      (coordinates, err) => {
        if (err) {
          setError('Nepodařilo se získat polohu. Zkontrolujte oprávnění a nastavení zařízení.');
          return;
        }
        if (coordinates) {
          setError(null);
          const pos: PositionData = {
            lat: coordinates.coords.latitude,
            lng: coordinates.coords.longitude,
            accuracy: coordinates.coords.accuracy,
            altitude: coordinates.coords.altitude,
            altitudeAccuracy: coordinates.coords.altitudeAccuracy,
            timestamp: coordinates.timestamp,
          };

          // Výpočet rychlosti a směru
          if (prevPos.current && pos.timestamp && prevPos.current.timestamp) {
            const dt = (pos.timestamp - prevPos.current.timestamp) / 1000; // sekundy
            if (dt > 0) {
              const dist = getDistance(prevPos.current.lat, prevPos.current.lng, pos.lat, pos.lng); // metry
              setSpeed(dist / dt); // m/s
              setBearing(getBearing(prevPos.current.lat, prevPos.current.lng, pos.lat, pos.lng));
            }
          }
          prevPos.current = pos;

          setPosition(pos);
          if (onPosition) {
            onPosition({ lat: pos.lat, lng: pos.lng });
          }
        }
      }
    ).then(id => {
      watchId = id;
    });

    return () => {
      if (watchId) {
        Geolocation.clearWatch({ id: watchId });
      }
    };
  }, [onPosition]);

  return (
    <IonCard>
      <IonCardContent style={{ textAlign: 'center' }}>
        {error && <IonText color="danger">{error}</IonText>}
        {position ? (
          <IonGrid>
            <IonRow>
              <IonCol>
                <IonText color="medium">Lat:</IonText>
                <div>{position.lat.toFixed(5)}</div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Lng:</IonText>
                <div>{position.lng.toFixed(5)}</div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Acc:</IonText>
                <div>{position.accuracy ?? 'N/A'} m</div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Alt:</IonText>
                <div>
                  {position.altitude !== undefined && position.altitude !== null
                    ? position.altitude.toFixed(2)
                    : 'N/A'} m
                </div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Alt acc:</IonText>
                <div>
                  {position.altitudeAccuracy !== undefined && position.altitudeAccuracy !== null
                    ? position.altitudeAccuracy.toFixed(2)
                    : 'N/A'} m
                </div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Rychlost:</IonText>
                <div>
                  {speed !== null ? speed.toFixed(2) + ' m/s' : 'N/A'}
                </div>
              </IonCol>
              <IonCol>
                <IonText color="medium">Směr:</IonText>
                <div>
                  {bearing !== null ? bearing.toFixed(0) + '°' : 'N/A'}
                </div>
              </IonCol>
            </IonRow>
          </IonGrid>
        ) : (
          <IonSpinner name="dots" />
        )}
      </IonCardContent>
    </IonCard>
  );
};

export default CurrentPosition;