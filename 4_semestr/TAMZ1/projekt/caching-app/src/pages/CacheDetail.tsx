import React, { useEffect, useState } from 'react';
import {
  IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonCard, IonCardHeader, IonCardTitle, IonCardContent,
  IonButton, IonList, IonItem, IonLabel, IonText, IonIcon
} from '@ionic/react';
import { useHistory } from 'react-router-dom';
import { Geolocation } from '@capacitor/geolocation';
import MapContainer from '../components/MapContainer';
import { pin, calendar, informationCircle, locate, walk } from 'ionicons/icons';

function getDistanceFromLatLonInKm(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6371;
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

const CacheDetail: React.FC = () => {
  const history = useHistory();
  const [cache, setCache] = useState<any>(null);
  const [distance, setDistance] = useState<string | null>(null);
  const [userPos, setUserPos] = useState<{ lat: number; lng: number } | null>(null);
  const [serverCache, setServerCache] = useState<any>(null);

  useEffect(() => {
    const localCache = JSON.parse(localStorage.getItem('selectedCache') || '{}');
    setCache(localCache);

    Geolocation.getCurrentPosition()
      .then(coords => {
        setUserPos({
          lat: coords.coords.latitude,
          lng: coords.coords.longitude,
        });
        if (localCache && localCache.lat && localCache.lng) {
          const dist = getDistanceFromLatLonInKm(
            coords.coords.latitude,
            coords.coords.longitude,
            localCache.lat,
            localCache.lng
          );
          setDistance(dist.toFixed(2));
        }
      })
      .catch(() => setUserPos(null));

    if (localCache && localCache.name) {
      fetch(`${process.env.REACT_APP_API_URL}/api/Caching/detail/${encodeURIComponent(localCache.name)}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      })
        .then(response => {
          if (response.ok) {
            return response.json();
          } else {
            console.error('Chyba při načítání detailu cache z API: Status', response.status);
            throw new Error('Neúspěšný požadavek na server');
          }
        })
        .then(data => {
          setServerCache(data);
        })
        .catch((error) => {
          console.error('Chyba při načítání detailu cache z API:', error);
        });
    }
  }, []);

  const cacheData = serverCache || cache;

  if (!cacheData || !cacheData.name) {
    return (
      <IonPage>
        <IonHeader>
          <IonToolbar>
            <IonTitle>Detail cache</IonTitle>
          </IonToolbar>
        </IonHeader>
        <IonContent fullscreen>
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>Cache nenalezena</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <IonButton onClick={() => history.goBack()}>Zpět</IonButton>
            </IonCardContent>
          </IonCard>
        </IonContent>
      </IonPage>
    );
  }

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Detail cache</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>
              <IonIcon icon={pin} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              {cacheData.name}
            </IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <IonList lines="none">
              <IonItem>
                <IonIcon icon={calendar} slot="start" />
                <IonLabel>Přidáno</IonLabel>
                <IonText slot="end">{cacheData.date ? new Date(cacheData.date).toLocaleString() : '-'}</IonText>
              </IonItem>
              <IonItem>
                <IonIcon icon={locate} slot="start" />
                <IonLabel>Pozice</IonLabel>
                <IonText slot="end">{cacheData.lat.toFixed(2)}, {cacheData.lng.toFixed(2)}</IonText>
              </IonItem>
              {distance && (
                <IonItem>
                  <IonIcon icon={walk} slot="start" />
                  <IonLabel>Vzdálenost od vás</IonLabel>
                  <IonText slot="end">{distance} km</IonText>
                </IonItem>
              )}
              {cacheData.description && (
                <IonItem>
                  <IonIcon icon={informationCircle} slot="start" />
                  <IonLabel>
                    <b>Popis</b>
                    <div style={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
                      {cacheData.description}
                    </div>
                  </IonLabel>
                </IonItem>
              )}
              {/* Další informace z API, např. owner, difficulty, size, hint, logs... */}
            </IonList>
            <MapContainer position={cacheData.lat && cacheData.lng ? { lat: cacheData.lat, lng: cacheData.lng, name: cacheData.name } : null} />
            <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
              <IonButton color="medium" onClick={() => {
                localStorage.setItem('navigateTo', JSON.stringify(cacheData));
                history.push('/navigate');
              }}>
                Navigovat
              </IonButton>
              <IonButton onClick={() => history.goBack()}>
                Zpět
              </IonButton>
            </div>
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default CacheDetail;