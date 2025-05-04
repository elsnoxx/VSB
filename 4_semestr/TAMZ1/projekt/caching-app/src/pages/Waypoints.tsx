import React, { useState, useEffect } from 'react';
import {
  IonContent,
  IonHeader,
  IonPage,
  IonTitle,
  IonToolbar,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
  IonList,
  IonItem,
  IonLabel,
  IonIcon,
  IonButton
} from '@ionic/react';
import { locationOutline } from 'ionicons/icons';
import { useHistory } from 'react-router-dom';
import WaypointsList from '../components/WaypointsList';

type Cache = {
  name: string;
  date: string;
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
};

const Caches: React.FC = () => {
  const history = useHistory();
  const [caches, setCaches] = useState<Cache[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCaches = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/Caching`);
      if (!response.ok) {
        setError(`Chyba serveru: ${response.status} ${response.statusText}`);
        setCaches([]);
        localStorage.removeItem('waypoints');
      } else {
        const data = await response.json();
        if (!data || data.length === 0) {
          setError('Ze serveru nebylo vráceno nic.');
          setCaches([]);
          localStorage.removeItem('waypoints');
        } else {
          setCaches(data);
          // Ulož do localStorage pro další použití v aplikaci
          localStorage.setItem('waypoints', JSON.stringify(data));
        }
      }
    } catch (error) {
      setError('Chyba při načítání cachek.');
      setCaches([]);
      localStorage.removeItem('waypoints');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchCaches();
  }, []);

  const handleSelect = (cache: Cache) => {
    localStorage.setItem('selectedCache', JSON.stringify(cache));
    history.push('/cache-detail');
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Uložené cache</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>
              <IonIcon icon={locationOutline} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Seznam cachek
            </IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <IonButton expand="block" onClick={fetchCaches} style={{ marginBottom: 8 }}>
              Aktualizovat seznam
            </IonButton>
            {loading ? (
              <div>Načítání...</div>
            ) : error ? (
              <div style={{ color: 'red' }}>{error}</div>
            ) : caches.length > 0 ? (
              <WaypointsList waypoints={caches} onSelect={handleSelect} />
            ) : null}
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default Caches;