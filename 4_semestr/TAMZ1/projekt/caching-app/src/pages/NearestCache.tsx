import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonButton, IonInput, IonItem, IonLabel, IonList } from '@ionic/react';
import { Geolocation } from '@capacitor/geolocation';
import { useHistory } from 'react-router-dom';

type Cache = {
  name: string;
  date: string;
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
};

const NearestCache: React.FC = () => {
  const [position, setPosition] = useState<{ lat: number; lng: number } | null>(null);
  const [radius, setRadius] = useState<number>(1);
  const [caches, setCaches] = useState<Cache[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const history = useHistory();

  useEffect(() => {
    const getCurrentPosition = async () => {
      try {
        const coordinates = await Geolocation.getCurrentPosition();
        setPosition({
          lat: coordinates.coords.latitude,
          lng: coordinates.coords.longitude
        });
      } catch {
        setError('Nepodařilo se získat polohu.');
      }
    };
    getCurrentPosition();
  }, []);

  const fetchNearbyCaches = async () => {
    if (!position) {
      setError('Poloha není k dispozici.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const url = `${process.env.REACT_APP_API_URL}/api/Caching/nearby?lat=${position.lat}&lng=${position.lng}&radius=${radius}`;
      const response = await fetch(url);
      if (!response.ok) {
        setError(`Chyba serveru: ${response.status} ${response.statusText}`);
        setCaches([]);
      } else {
        const data = await response.json();
        if (!data || data.length === 0) {
          setError('Žádné cache v okolí.');
          setCaches([]);
        } else {
          setCaches(data);
        }
      }
    } catch {
      setError('Chyba při načítání cachek.');
      setCaches([]);
    }
    setLoading(false);
  };

  // Pomocná funkce pro výpočet vzdálenosti mezi dvěma body (Haversine)
  function getDistanceFromLatLonInKm(lat1: number, lon1: number, lat2: number, lon2: number) {
    const R = 6371; // Radius of the earth in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  const handleCacheClick = (cache: Cache) => {
    // Uložení detailu do localStorage nebo jinam, případně přesměrování s parametrem
    localStorage.setItem('selectedCache', JSON.stringify(cache));
    history.push('/cache-detail');
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Nejbližší cache</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>Vyhledat cache v okolí</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <IonItem>
              <IonLabel position="stacked">Poloměr hledání (km)</IonLabel>
              <IonInput
                type="number"
                value={radius}
                min="0.1"
                step="0.1"
                onIonChange={e => setRadius(Number(e.detail.value) || 1)}
              />
            </IonItem>
            <IonButton expand="block" onClick={fetchNearbyCaches} disabled={!position || loading}>
              {loading ? 'Načítání...' : 'Najít cache'}
            </IonButton>
            {error && <div style={{ color: 'red', marginTop: 8 }}>{error}</div>}
            {caches.length > 0 && (
              <IonList>
                {caches.map((cache, idx) => (
                  <IonItem
                    key={idx}
                    button
                    detail
                    onClick={() => handleCacheClick(cache)}
                  >
                    <IonLabel>
                      <b>{cache.name}</b>
                      <div>
                        {cache.lat}, {cache.lng}
                        {position && (
                          <>
                            <br />
                            Vzdálenost:{' '}
                            {getDistanceFromLatLonInKm(
                              position.lat,
                              position.lng,
                              cache.lat,
                              cache.lng
                            ).toFixed(2)}{' '}
                            km
                          </>
                        )}
                        <br />
                        {cache.date}
                      </div>
                    </IonLabel>
                  </IonItem>
                ))}
              </IonList>
            )}
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default NearestCache;