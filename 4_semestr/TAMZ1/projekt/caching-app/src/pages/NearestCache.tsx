import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonButton, IonInput, IonItem, IonLabel, IonList, IonInfiniteScroll, IonInfiniteScrollContent, IonBadge } from '@ionic/react';
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

const PAGE_SIZE = 10;

const NearestCache: React.FC = () => {
  const [position, setPosition] = useState<{ lat: number; lng: number } | null>(null);
  const [radius, setRadius] = useState<number>(1);
  const [allCaches, setAllCaches] = useState<Cache[]>([]);
  const [displayedCaches, setDisplayedCaches] = useState<Cache[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const history = useHistory();

  // Přidejte toto pro získání názvů nalezených keší
  const foundCaches = JSON.parse(localStorage.getItem('foundCaches') || '[]') as { name: string }[];
  const foundNames = foundCaches.map(fc => fc.name);

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
    setAllCaches([]); // smaž stará data
    setDisplayedCaches([]);
    setPage(1);
    setHasMore(true);
    
    try {
      // Použijeme standardní fetch místo Capacitor HTTP
      // Vytvoření URL s query parametry
      const url = new URL(`${process.env.REACT_APP_API_URL}/api/Caching/nearby`);
      url.searchParams.append('lat', position.lat.toString());
      url.searchParams.append('lng', position.lng.toString());
      url.searchParams.append('radius', radius.toString());
      
      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });

      // Kontrola úspěšné odpovědi
      if (!response.ok) {
        setError(`Chyba serveru: ${response.status}`);
        setAllCaches([]);
        setDisplayedCaches([]);
        setHasMore(false);
      } else {
        // Zpracování dat - musíme explicitně parsovat JSON
        const data = await response.json();
        
        if (!data || (Array.isArray(data) && data.length === 0)) {
          setError('Žádné cache v okolí.');
          setAllCaches([]);
          setDisplayedCaches([]);
          setHasMore(false);
        } else {
          const caches = Array.isArray(data) ? data : [data];
          setAllCaches(caches);
          setDisplayedCaches(caches.slice(0, PAGE_SIZE));
          setHasMore(caches.length > PAGE_SIZE);
        }
      }
    } catch (error: any) {
      setError('Chyba při načítání cachek: ' + (error?.message || 'Neznámá chyba'));
      console.error('HTTP chyba:', error);
      setAllCaches([]);
      setDisplayedCaches([]);
      setHasMore(false);
    }
    
    setLoading(false);
  };

  const loadMore = (ev: CustomEvent<void>) => {
    const nextPage = page + 1;
    const nextItems = allCaches.slice(0, nextPage * PAGE_SIZE);
    setDisplayedCaches(nextItems);
    setPage(nextPage);
    setHasMore(nextItems.length < allCaches.length);
    (ev.target as HTMLIonInfiniteScrollElement).complete();
  };

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

  const handleCacheClick = (cache: Cache) => {
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
            {displayedCaches.length > 0 && (
              <IonList>
                {displayedCaches.map((cache, idx) => (
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
                    {/* Zobraz badge pokud je cache nalezena */}
                    {foundNames.includes(cache.name) && (
                      <IonBadge color="success" slot="end">
                        Nalezeno
                      </IonBadge>
                    )}
                  </IonItem>
                ))}
                <IonInfiniteScroll
                  threshold="100px"
                  disabled={!hasMore}
                  onIonInfinite={loadMore}
                >
                  <IonInfiniteScrollContent loadingText="Načítání dalších cache..." />
                </IonInfiniteScroll>
              </IonList>
            )}
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default NearestCache;