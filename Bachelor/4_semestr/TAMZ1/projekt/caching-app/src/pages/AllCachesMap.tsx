import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonCard, IonCardHeader, IonCardTitle, IonButton, IonIcon, IonModal, IonCardContent, IonList, IonItem, IonLabel, IonBadge } from '@ionic/react';
import { arrowBack, checkmarkCircle } from 'ionicons/icons';
import MapContainerMultiple from '../components/MapContainerMultiple';
import { useHistory } from 'react-router-dom';

type Cache = {
  name?: string;
  lat: number;
  lng: number;
  found?: boolean;
};

const AllCachesMap: React.FC = () => {
  const [caches, setCaches] = useState<Cache[]>([]);
  const [myPosition, setMyPosition] = useState<{ lat: number; lng: number } | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<any>(null);
  const [showModal, setShowModal] = useState(false);
  const [foundCacheNames, setFoundCacheNames] = useState<string[]>([]);
  const history = useHistory();

  // Načtení waypointů a nalezených kešek
  useEffect(() => {
    // Načti waypointy z localStorage
    const local = localStorage.getItem('waypoints');
    if (local) {
      try {
        setCaches(JSON.parse(local));
      } catch {
        setCaches([]);
      }
    } else {
      setCaches([]);
    }
    
    // Načti seznam nalezených kešek
    try {
      const found = localStorage.getItem('foundCaches');
      if (found) {
        const foundCaches = JSON.parse(found);
        // Vytvoř pole názvů nalezených kešek
        setFoundCacheNames(foundCaches.map((cache: any) => cache.name));
      }
    } catch (e) {
      console.error('Chyba načtení nalezených kešek', e);
    }
  }, []);

  // Získání aktuální polohy
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      pos => setMyPosition({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => setMyPosition(null)
    );
  }, []);

  const handleMarkerClick = async (cache: Cache) => {
    // Přidejte toto pro ladění:
    console.log("Volána funkce handleMarkerClick s daty:", cache);
    
    try {
      // Zkontrolovat, zda cache má name
      if (!cache.name) {
        console.log("Cache nemá jméno, zobrazuji základní info");
        setSelectedDetail({
          ...cache,
          found: cache.found || false
        });
        setShowModal(true);
        return;
      }
      
      // Nejprve nastavte základní detail, aby se modál mohl zobrazit i při čekání na API
      setSelectedDetail({
        ...cache,
        found: cache.found || foundCacheNames.includes(cache.name)
      });
      setShowModal(true);
      
      // Pak se pokuste načíst více detailů pomocí standardního fetch API
      try {
        const response = await fetch(
          `${process.env.REACT_APP_API_URL}/api/Caching/detail/${encodeURIComponent(cache.name)}`,
          {
            method: 'GET',
            headers: {
              'Accept': 'application/json'
            }
          }
        );
        
        if (response.ok) {
          // Musíme rozparsovat JSON odpověď
          const data = await response.json();
          setSelectedDetail({
            ...data,
            found: cache.found || foundCacheNames.includes(cache.name)
          });
        } else {
          console.error('Chyba při získávání detailu kešky:', response.status);
        }
      } catch (e) {
        console.error('Chyba načtení detailu kešky', e);
      }
    } catch (e) {
      console.error('Chyba zpracování kliknutí', e);
    }
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonButton slot="start" fill="clear" onClick={() => history.goBack()}>
            <IonIcon icon={arrowBack} />
            Zpět
          </IonButton>
          <IonTitle>Všechny cache na mapě</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <div style={{ width: '100vw', maxWidth: '100vw', height: '200vh', margin: 0, padding: 0 }}>
          <MapContainerMultiple
            caches={caches}
            onMarkerClick={handleMarkerClick}
            myPosition={myPosition}
            foundCacheNames={foundCacheNames}
          />
        </div>
        <IonModal isOpen={showModal} onDidDismiss={() => setShowModal(false)}>
          <IonCardHeader>
            <IonCardTitle>
              Detail kešky
              {selectedDetail?.found && (
                <IonBadge color="success" style={{ marginLeft: 10 }}>
                  <IonIcon icon={checkmarkCircle} /> Nalezeno
                </IonBadge>
              )}  
            </IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            {selectedDetail ? (
              <IonList>
                <IonItem>
                  <IonLabel><b>Název:</b> {selectedDetail.name}</IonLabel>
                </IonItem>
                <IonItem>
                  <IonLabel><b>Souřadnice:</b> {selectedDetail.lat}, {selectedDetail.lng}</IonLabel>
                </IonItem>
                {/* Zobrazit další informace, pokud jsou dostupné */}
                {selectedDetail.description && (
                  <IonItem>
                    <IonLabel><b>Popis:</b> {selectedDetail.description}</IonLabel>
                  </IonItem>
                )}
                {selectedDetail.found && (
                  <IonItem>
                    <IonLabel><b>Stav:</b> Nalezeno</IonLabel>
                  </IonItem>
                )}
              </IonList>
            ) : (
              <div>Načítání detailu...</div>
            )}
            <IonButton expand="block" onClick={() => setShowModal(false)}>Zavřít</IonButton>
            
            {/* Přidat tlačítko pro navigaci ke kešce */}
            {selectedDetail && (
              <IonButton 
                expand="block" 
                color="primary" 
                onClick={() => {
                  if (selectedDetail) {
                    localStorage.setItem('navigateTo', JSON.stringify({
                      name: selectedDetail.name,
                      lat: selectedDetail.lat,
                      lng: selectedDetail.lng,
                      found: selectedDetail.found
                    }));
                    history.push('/navigate');
                  }
                }}
                className="ion-margin-top"
              >
                Navigovat k této kešce
              </IonButton>
            )}
          </IonCardContent>
        </IonModal>
      </IonContent>
    </IonPage>
  );
};

export default AllCachesMap;