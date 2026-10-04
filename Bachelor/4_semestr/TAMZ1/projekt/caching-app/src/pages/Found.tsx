import React, { useEffect, useState } from 'react';
import { 
  IonPage, 
  IonHeader, 
  IonToolbar, 
  IonTitle, 
  IonContent, 
  IonButtons, 
  IonBackButton, 
  IonCard, 
  IonCardHeader, 
  IonCardTitle, 
  IonCardContent, 
  IonList, 
  IonItem, 
  IonLabel, 
  IonIcon, 
  IonBadge, 
  IonInfiniteScroll, 
  IonInfiniteScrollContent, 
  IonSearchbar, 
  IonSegment, 
  IonSegmentButton, 
  IonButton 
} from '@ionic/react';
import { checkmarkCircle, calendar, person, location, navigate, trophy, downloadOutline } from 'ionicons/icons';
import MapContainerMultiple from '../components/MapContainerMultiple';

// Typ pro nalezenou kešku
type FoundCache = {
  name: string;
  foundAt: string;
  foundBy: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  distance?: string;
};

const Found: React.FC = () => {
  const [allFoundCaches, setAllFoundCaches] = useState<FoundCache[]>([]);
  const [displayedCaches, setDisplayedCaches] = useState<FoundCache[]>([]);
  const [totalFound, setTotalFound] = useState<number>(0);
  const [mostRecentCache, setMostRecentCache] = useState<FoundCache | null>(null);
  const [page, setPage] = useState<number>(1);
  const PAGE_SIZE = 10;

  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterType, setFilterType] = useState<string>('all');
  
  useEffect(() => {
    // Načti nalezené kešky z localStorage
    const cacheData = localStorage.getItem('foundCaches');
    if (cacheData) {
      try {
        const caches = JSON.parse(cacheData);
        // Seřaď kešky podle data nalezení (nejnovější první)
        caches.sort((a: FoundCache, b: FoundCache) => 
          new Date(b.foundAt).getTime() - new Date(a.foundAt).getTime()
        );
        
        setAllFoundCaches(caches);
        setDisplayedCaches(caches.slice(0, PAGE_SIZE));
        setTotalFound(caches.length);
        
        // Nastav nejnovější kešku
        if (caches.length > 0) {
          setMostRecentCache(caches[0]);
        }
      } catch (e) {
        console.error('Chyba při načítání nalezených kešek:', e);
        setAllFoundCaches([]);
        setDisplayedCaches([]);
      }
    }
  }, []);

  // Formátovač data
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('cs-CZ', {
      day: '2-digit',
      month: '2-digit', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  // Načíst další stránku při scrollování
  const loadMoreData = (event: CustomEvent<void>) => {
    setTimeout(() => {
      const nextPage = page + 1;
      const startIndex = (nextPage - 1) * PAGE_SIZE;
      const endIndex = nextPage * PAGE_SIZE;
      
      // Přidej další položky do zobrazeného seznamu
      const newItems = allFoundCaches
        .filter(cache => 
          cache.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
          (filterType === 'all' ||
          (filterType === 'week' && new Date(cache.foundAt) >= new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)) ||
          (filterType === 'month' && new Date(cache.foundAt) >= new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)))
        )
        .slice(startIndex, endIndex);
      setDisplayedCaches([...displayedCaches, ...newItems]);
      setPage(nextPage);
      
      // Ukonči event nekonečného scrollování
      (event.target as HTMLIonInfiniteScrollElement).complete();
      
      // Pokud už nejsou další položky, deaktivuj infinite scroll
      if (displayedCaches.length >= allFoundCaches.length) {
        (event.target as HTMLIonInfiniteScrollElement).disabled = true;
      }
    }, 500); // Simulujeme načítání pro lepší UX
  };

  const statistics = {
    totalFound: totalFound,
    thisWeek: allFoundCaches.filter(c => {
      const cacheDate = new Date(c.foundAt);
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
      return cacheDate >= weekAgo;
    }).length,
    thisMonth: allFoundCaches.filter(c => {
      const cacheDate = new Date(c.foundAt);
      const monthAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
      return cacheDate >= monthAgo;
    }).length
  };

  // Přidat funkce pro export/import
  const exportData = () => {
    const data = localStorage.getItem('foundCaches');
    const blob = new Blob([data || '[]'], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `geocaching-export-${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar color="primary">
          <IonButtons slot="start">
            <IonBackButton defaultHref="/home" />
          </IonButtons>
          <IonTitle>Nalezené kešky</IonTitle>
          <IonButtons slot="end">
            <IonButton onClick={exportData}>
              <IonIcon slot="icon-only" icon={downloadOutline} />
            </IonButton>
            <IonBadge color="light" className="ion-margin-end">
              <IonIcon icon={trophy} /> {totalFound}
            </IonBadge>
          </IonButtons>
        </IonToolbar>
        <IonToolbar>
          <IonSearchbar 
            value={searchTerm}
            onIonChange={e => setSearchTerm(e.detail.value!)}
            placeholder="Vyhledat kešku..."
          />
          <IonSegment value={filterType} onIonChange={e => setFilterType(e.detail.value as string)}>
            <IonSegmentButton value="all">
              <IonLabel>Vše</IonLabel>
            </IonSegmentButton>
            <IonSegmentButton value="week">
              <IonLabel>Týden</IonLabel>
            </IonSegmentButton>
            <IonSegmentButton value="month">
              <IonLabel>Měsíc</IonLabel>
            </IonSegmentButton>
          </IonSegment>
        </IonToolbar>
      </IonHeader>
      
      <IonContent fullscreen>
        {totalFound > 0 ? (
          <>
            {/* Mapa s nalezenými keškami */}
            <div style={{ position: 'relative', height: '200px', width: '100%', margin: '0 0 16px 0' }}>
              <MapContainerMultiple 
                caches={allFoundCaches.map(fc => ({
                  name: fc.name,
                  lat: fc.coordinates.lat,
                  lng: fc.coordinates.lng,
                  found: true
                }))}
              />
            </div>

            {/* Shrnutí */}
            <IonCard className="ion-margin-horizontal">
              <IonCardHeader>
                <IonCardTitle>
                  <IonIcon icon={trophy} color="warning" /> Statistika
                </IonCardTitle>
              </IonCardHeader>
              <IonCardContent>
                <p>Celkem nalezeno: <strong>{totalFound} kešek</strong></p>
                <p>Tento týden: <strong>{statistics.thisWeek}</strong></p>
                <p>Tento měsíc: <strong>{statistics.thisMonth}</strong></p>
                {mostRecentCache && (
                  <p>Poslední nález: <strong>{mostRecentCache.name}</strong> ({formatDate(mostRecentCache.foundAt)})</p>
                )}
                
                {/* Odznaky */}
                {totalFound >= 1 && 
                  <IonBadge color="success" className="ion-margin-end">První nález</IonBadge>
                }
                {totalFound >= 5 && 
                  <IonBadge color="warning" className="ion-margin-end">Sběratel (5+)</IonBadge>
                }
                {totalFound >= 10 && 
                  <IonBadge color="tertiary">Průzkumník (10+)</IonBadge>
                }
              </IonCardContent>
            </IonCard>

            {/* Seznam nalezených kešek - nekonečný seznam */}
            <IonCard className="ion-margin-horizontal">
              <IonCardHeader>
                <IonCardTitle>Historie nálezů</IonCardTitle>
              </IonCardHeader>
              <IonList>
                {displayedCaches.map((cache, index) => (
                  <IonItem key={index}>
                    <IonIcon icon={checkmarkCircle} slot="start" color="success" />
                    <IonLabel>
                      <h2>{cache.name}</h2>
                      <p>
                        <IonIcon icon={calendar} /> {formatDate(cache.foundAt)}
                      </p>
                      <p>
                        <IonIcon icon={person} /> {cache.foundBy}
                      </p>
                      <p>
                        <IonIcon icon={location} /> {cache.coordinates.lat.toFixed(6)}, {cache.coordinates.lng.toFixed(6)}
                      </p>
                      {cache.distance && (
                        <p>
                          <IonIcon icon={navigate} /> Vzdálenost při nálezu: {cache.distance} m
                        </p>
                      )}
                    </IonLabel>
                  </IonItem>
                ))}
              </IonList>
              
              {/* Nekonečný scroll */}
              <IonInfiniteScroll threshold="100px" onIonInfinite={loadMoreData}>
                <IonInfiniteScrollContent
                  loadingSpinner="bubbles"
                  loadingText="Načítám další kešky..."
                >
                </IonInfiniteScrollContent>
              </IonInfiniteScroll>
            </IonCard>
          </>
        ) : (
          <IonCard className="ion-margin">
            <IonCardHeader>
              <IonCardTitle>Žádné nalezené kešky</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <p>Zatím jste nenašli žádnou kešku. Použijte navigaci a vydejte se na dobrodružství!</p>
            </IonCardContent>
          </IonCard>
        )}
      </IonContent>
    </IonPage>
  );
};

export default Found;