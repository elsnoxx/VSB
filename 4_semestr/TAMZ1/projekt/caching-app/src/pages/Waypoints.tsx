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
  IonButton,
  IonRow,
  IonCol,
  IonBadge,
  IonInfiniteScroll,
  IonInfiniteScrollContent
} from '@ionic/react';
import { locationOutline, checkmarkCircle } from 'ionicons/icons';
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

const PAGE_SIZE = 10;

const getStatusText = (status: number): string => {
  const statusTexts: Record<number, string> = {
    400: 'Špatný požadavek',
    401: 'Neautorizováno',
    403: 'Přístup odepřen',
    404: 'Nenalezeno',
    500: 'Interní chyba serveru',
    502: 'Špatná brána',
    503: 'Služba nedostupná',
    504: 'Časový limit brány vypršel'
  };
  return statusTexts[status] || 'Neznámá chyba';
};

const Caches: React.FC = () => {
  const history = useHistory();
  const [caches, setCaches] = useState<Cache[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [displayedCaches, setDisplayedCaches] = useState<Cache[]>([]);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  const foundCaches = JSON.parse(localStorage.getItem('foundCaches') || '[]') as { name: string }[];
  const foundNames = foundCaches.map(fc => fc.name);

  const fetchCaches = async () => {
    setLoading(true);
    setError(null);
    setErrorDetail(null);

    // Možnost dynamické změny URL (např. z localStorage)
    const apiUrl =
      localStorage.getItem('apiUrl') ||
      process.env.REACT_APP_API_URL ||
      '';

    if (!apiUrl) {
      setError('API adresa není nastavena.');
      setLoading(false);
      return;
    }

    try {
      // Použij standardní fetch místo Capacitor HTTP
      const response = await fetch(`${apiUrl}/api/Caching`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });

      // Kontrola úspěšné odpovědi (status 200)
      if (!response.ok) {
        setError(`Chyba serveru: ${response.status} - ${getStatusText(response.status)}`);
        let errorText = '';
        try {
          errorText = await response.text();
        } catch (e) {
          errorText = 'Nepodařilo se načíst detaily chyby';
        }
        setErrorDetail(errorText);
        setCaches([]);
        localStorage.removeItem('waypoints');
        console.error('Chyba serveru:', response.status, errorText);
      } else {
        // Data musíš zparsovat - fetch neparse data automaticky
        const data = await response.json();
        
        if (!data || (Array.isArray(data) && data.length === 0)) {
          setError('Ze serveru nebylo vráceno nic.');
          setCaches([]);
          localStorage.removeItem('waypoints');
        } else {
          setCaches(Array.isArray(data) ? data : [data]);
          localStorage.setItem('waypoints', JSON.stringify(data));
        }
      }
    } catch (error: any) {
      setError('Chyba komunikace se serverem: ' + (error?.message || 'Neznámá chyba'));
      setErrorDetail(JSON.stringify(error));
      setCaches([]);
      localStorage.removeItem('waypoints');
      console.error('HTTP chyba:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchCaches();
  }, []);

  useEffect(() => {
    setDisplayedCaches(caches.slice(0, PAGE_SIZE));
    setPage(1);
    setHasMore(caches.length > PAGE_SIZE);
  }, [caches]);

  const handleSelect = (cache: Cache) => {
    localStorage.setItem('selectedCache', JSON.stringify(cache));
    history.push('/cache-detail');
  };

  const loadMore = (ev: CustomEvent<void>) => {
    const nextPage = page + 1;
    const nextItems = caches.slice(0, nextPage * PAGE_SIZE);
    setDisplayedCaches(nextItems);
    setPage(nextPage);
    setHasMore(nextItems.length < caches.length);
    (ev.target as HTMLIonInfiniteScrollElement).complete();
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
            <IonRow style={{ marginBottom: 8 }}>
              <IonCol>
                <IonButton expand="block" onClick={fetchCaches}>
                  Aktualizovat seznam
                </IonButton>
              </IonCol>
              <IonCol>
                <IonButton expand="block" onClick={() => history.push('/all-caches-map')}>
                  Mapa
                </IonButton>
              </IonCol>
            </IonRow>
            
            {loading ? (
              <div>Načítání...</div>
            ) : error ? (
              <div style={{ color: 'red' }}>{error}</div>
            ) : caches.length > 0 ? (
              <IonList>
                {displayedCaches.map((cache, idx) => (
                  <IonItem key={idx} button onClick={() => handleSelect(cache)}>
                    <IonLabel>
                      <h2>{cache.name}</h2>
                      <p>{cache.lat}, {cache.lng}</p>
                    </IonLabel>
                    {foundNames.includes(cache.name) && (
                      <IonBadge color="success" slot="end">
                        <IonIcon icon={checkmarkCircle} style={{ marginRight: 4 }} />
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
            ) : null}
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default Caches;