import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonCard, IonCardHeader, IonCardTitle } from '@ionic/react';
import MapContainer from '../components/MapContainer';

type Cache = {
  name: string;
  lat: number;
  lng: number;
  // případně další pole
};

const AllCachesMap: React.FC = () => {
  const [caches, setCaches] = useState<Cache[]>([]);

  useEffect(() => {
    // Načtení všech cache z API
    fetch(`${process.env.REACT_APP_API_URL}/api/Caching`)
      .then(res => res.ok ? res.json() : [])
      .then(data => setCaches(data))
      .catch(() => setCaches([]));
  }, []);

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Všechny cache na mapě</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>Mapa všech dostupných cachek</IonCardTitle>
          </IonCardHeader>
          <MapContainer
            // předáme pole všech pozic, upravte MapContainer aby podporoval více markerů
            positions={caches.map(c => ({ lat: c.lat, lng: c.lng, name: c.name }))}
          />
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default AllCachesMap;