import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonItem, IonLabel, IonSelect, IonSelectOption, IonList, IonButton, IonCard, IonCardHeader, IonCardTitle, IonCardContent } from '@ionic/react';
import CompassComponent from '../components/CompassComponent';
import CurrentPosition from '../components/CurrentPosition';
import DualMapContainer from '../components/DualMapContainer';

type Waypoint = {
  name: string;
  lat: number;
  lng: number;
};

const getWaypoints = (): Waypoint[] => {
  const data = localStorage.getItem('waypoints');
  return data ? JSON.parse(data) : [];
};

const Navigate: React.FC = () => {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);
  const [selected, setSelected] = useState<Waypoint | null>(null);
  const [currentPos, setCurrentPos] = useState<{ lat: number; lng: number } | null>(null);

  useEffect(() => {
    setWaypoints(getWaypoints());
  }, []);

  useEffect(() => {
    const nav = localStorage.getItem('navigateTo');
    if (nav) {
      const navPoint = JSON.parse(nav);
      setSelected(navPoint);
      localStorage.removeItem('navigateTo');
    }
  }, []);

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Kompas</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        {!selected ? (
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>Vyberte referenční bod</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <IonItem>
                <IonLabel>Poloha</IonLabel>
                <IonSelect
                  placeholder="-- Vyberte bod --"
                  onIonChange={e => {
                    const idx = Number(e.detail.value);
                    if (!isNaN(idx)) setSelected(waypoints[idx]);
                  }}
                  value=""
                >
                  {waypoints.map((wp, i) => (
                    <IonSelectOption key={i} value={i}>
                      {wp.name} ({wp.lat.toFixed(2)}, {wp.lng.toFixed(2)})
                    </IonSelectOption>
                  ))}
                </IonSelect>
              </IonItem>
              {waypoints.length === 0 && (
                <div style={{ color: 'red', textAlign: 'center', marginTop: 16 }}>
                  Žádné body nejsou uloženy.
                </div>
              )}
            </IonCardContent>
          </IonCard>
        ) : (
          <IonList>
            <IonItem>
              <IonLabel>
                <b>Referenční bod:</b> {selected.name} ({selected.lat.toFixed(2)}, {selected.lng.toFixed(2)})
              </IonLabel>
            </IonItem>
            <IonItem>
              <DualMapContainer current={currentPos} target={selected} />
            </IonItem>
            <IonItem>
              <CurrentPosition onPosition={setCurrentPos} />
            </IonItem>
            <IonItem>
              {currentPos ? (
                <CompassComponent target={selected} currentPosition={currentPos} />
              ) : (
                <IonLabel color="danger">Čekám na určení aktuální polohy…</IonLabel>
              )}
            </IonItem>
            <IonButton expand="block" color="medium" onClick={() => setSelected(null)} style={{ marginTop: 16 }}>
              Změnit referenční bod
            </IonButton>
          </IonList>
        )}
      </IonContent>
    </IonPage>
  );
};

export default Navigate;