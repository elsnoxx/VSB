import React, { useEffect, useState } from 'react';
import { IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonText } from '@ionic/react';

declare global {
  interface Navigator {
    compass?: {
      watchHeading(
        successCallback: (headingData: { magneticHeading: number }) => void,
        errorCallback: (error: any) => void,
        options?: { frequency?: number }
      ): number;
      clearWatch(watchId: number): void;
    };
  }
}

type Waypoint = {
  name: string;
  lat: number;
  lng: number;
};

interface CompassComponentProps {
  target: Waypoint;
  currentPosition: { lat: number; lng: number };
}

const CompassComponent: React.FC<CompassComponentProps> = ({ target, currentPosition }) => {
  const [heading, setHeading] = useState<number | null>(null);
  const [compassAvailable, setCompassAvailable] = useState<boolean>(true);

  useEffect(() => {
    let watchId: number | null = null;

    if (window.navigator.compass) {
      watchId = window.navigator.compass.watchHeading(
        (headingData) => {
          setHeading(headingData.magneticHeading);
        },
        (error) => {
          setHeading(null);
        },
        { frequency: 500 }
      );
    } else {
      setCompassAvailable(false);
    }

    return () => {
      if (window.navigator.compass && watchId !== null) {
        window.navigator.compass.clearWatch(watchId);
      }
    };
  }, []);

  function getDistance(lat1: number, lon1: number, lat2: number, lon2: number) {
    const R = 6371e3;
    const φ1 = lat1 * Math.PI/180;
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    return R * c;
  }

  function getBearing(lat1: number, lon1: number, lat2: number, lon2: number) {
    const toRad = (deg: number) => deg * Math.PI / 180;
    const toDeg = (rad: number) => rad * 180 / Math.PI;
    const φ1 = toRad(lat1);
    const φ2 = toRad(lat2);
    const Δλ = toRad(lon2 - lon1);

    const y = Math.sin(Δλ) * Math.cos(φ2);
    const x = Math.cos(φ1) * Math.sin(φ2) -
              Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);
    const θ = Math.atan2(y, x);
    return (toDeg(θ) + 360) % 360;
  }

  // Výpočty pro zobrazení
  const bearing = getBearing(
    currentPosition.lat,
    currentPosition.lng,
    target.lat,
    target.lng
  );
  const distance = getDistance(
    currentPosition.lat,
    currentPosition.lng,
    target.lat,
    target.lng
  );
  const turn = heading !== null ? ((bearing - heading + 360) % 360) : null;

  return (
    <IonCard>
      <IonCardHeader>
        <IonCardTitle>Kompas</IonCardTitle>
      </IonCardHeader>
      <IonCardContent style={{ textAlign: 'center' }}>
        {!compassAvailable ? (
          <IonText color="danger">
            Kompas není na tomto zařízení dostupný nebo není podporován.
          </IonText>
        ) : heading !== null ? (
          <>
            <IonText color="medium">
              <div>Azimut zařízení: {heading.toFixed(0)}°</div>
              <div>Pochodový úhel (k cíli): {bearing.toFixed(0)}°</div>
              <div>Otoč se o: {turn !== null ? turn.toFixed(0) : '...'}°</div>
              <div>Vzdálenost k cíli: {distance.toFixed(1)} m</div>
            </IonText>
            <div
              style={{
                margin: '24px auto',
                width: 100,
                height: 100,
                border: '2px solid #333',
                borderRadius: '50%',
                position: 'relative',
              }}
            >
              {/* Červená ručička - aktuální směr zařízení */}
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  width: 2,
                  height: 40,
                  background: 'red',
                  transform: `translate(-50%, -100%) rotate(${heading}deg)`,
                  transformOrigin: 'bottom center',
                  zIndex: 2,
                }}
              />
              {/* Zelená ručička - směr k cíli (absolutní bearing) */}
              <div
                style={{
                  position: 'absolute',
                  left: '50%',
                  top: '50%',
                  width: 2,
                  height: 40,
                  background: 'green',
                  transform: `translate(-50%, -100%) rotate(${bearing}deg)`,
                  transformOrigin: 'bottom center',
                  zIndex: 1,
                }}
              />
            </div>
          </>
        ) : (
          <IonText>Načítám směr...</IonText>
        )}
      </IonCardContent>
    </IonCard>
  );
};

export default CompassComponent;