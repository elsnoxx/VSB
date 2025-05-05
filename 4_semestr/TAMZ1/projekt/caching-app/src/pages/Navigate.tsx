import React, { useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonItem, IonLabel, IonSelect, IonSelectOption, IonList, IonButton, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonToast, IonIcon, IonBadge, IonModal, IonInput } from '@ionic/react';
import CompassComponent from '../components/CompassComponent';
import CurrentPosition from '../components/CurrentPosition';
import DualMapContainer from '../components/DualMapContainer';
import MapContainer from '../components/MapContainer';
import { checkmarkCircle, alertCircle, person } from 'ionicons/icons';
import { useHistory } from 'react-router-dom';

type Waypoint = {
  name: string;
  lat: number;
  lng: number;
  found?: boolean;
};

const getWaypoints = (): Waypoint[] => {
  const data = localStorage.getItem('waypoints');
  return data ? JSON.parse(data) : [];
};

const Navigate: React.FC = () => {
  const [waypoints, setWaypoints] = useState<Waypoint[]>([]);
  const [selected, setSelected] = useState<Waypoint | null>(null);
  const [currentPos, setCurrentPos] = useState<{ lat: number; lng: number } | null>(null);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [isSuccess, setIsSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showNameModal, setShowNameModal] = useState(false);
  const [personName, setPersonName] = useState('');
  const [calculatedDistance, setCalculatedDistance] = useState<number>(0);
  const history = useHistory();

  useEffect(() => {
    setWaypoints(getWaypoints());
    const savedName = localStorage.getItem('personName');
    if (savedName) {
      setPersonName(savedName);
    }
  }, []);

  useEffect(() => {
    const nav = localStorage.getItem('navigateTo');
    if (nav) {
      const navPoint = JSON.parse(nav);
      setSelected(navPoint);
      localStorage.removeItem('navigateTo');
    }
  }, []);

  const handleMarkerClick = (wp: Waypoint) => {
    setSelected(wp);
  };

  const calculateDistance = (lat1: number, lng1: number, lat2: number, lng2: number) => {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLng = ((lng2 - lng1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLng / 2) * Math.sin(dLng / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c * 1000;
  };

  const handleOpenNameModal = () => {
    if (!selected || !currentPos) return;
    const distance = calculateDistance(currentPos.lat, currentPos.lng, selected.lat, selected.lng);
    setCalculatedDistance(distance);
    setShowNameModal(true);
  };

  const markAsFound = async () => {
    if (!selected || !currentPos || !personName.trim()) return;

    setIsSubmitting(true);
    setShowNameModal(false);

    try {
      // Uložíme jméno hledače
      localStorage.setItem('personName', personName);

      const foundDate = new Date();
      const foundDateISO = foundDate.toISOString();
      
      // Odeslání informace na server
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/Caching/found?cacheName=${encodeURIComponent(selected.name)}&personName=${encodeURIComponent(personName)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          cacheName: selected.name,
          personName: personName,
          foundAt: foundDateISO,
          distance: calculatedDistance.toFixed(2)
        })
      });

      if (response.ok) {
        // Aktualizace seznamu waypointů - příznak found
        const updatedWaypoints = waypoints.map(wp => 
          wp.name === selected.name ? { ...wp, found: true } : wp
        );
        setWaypoints(updatedWaypoints);
        localStorage.setItem('waypoints', JSON.stringify(updatedWaypoints));
        
        // Aktualizace vybraného waypointu
        setSelected({ ...selected, found: true });

        // NOVÉ: Uložení záznamu o nalezení do localStorage
        const foundCachesStr = localStorage.getItem('foundCaches') || '[]';
        const foundCaches = JSON.parse(foundCachesStr);
        
        // Přidání nového záznamu s názvem kešky a datem nalezení
        foundCaches.push({
          name: selected.name,
          foundAt: foundDateISO,
          foundBy: personName,
          coordinates: { lat: selected.lat, lng: selected.lng },
          distance: calculatedDistance.toFixed(2)
        });
        
        // Uložíme aktualizovaný seznam nalezených keší
        localStorage.setItem('foundCaches', JSON.stringify(foundCaches));

        setToastMessage(`Keška "${selected.name}" byla označena jako nalezená uživatelem ${personName}!`);
        setIsSuccess(true);
      } else {
        setToastMessage('Nastala chyba při označování kešky.');
        setIsSuccess(false);
      }
    } catch (error) {
      setToastMessage('Chyba připojení k serveru.');
      setIsSuccess(false);
    } finally {
      setIsSubmitting(false);
      setShowToast(true);
    }
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Kompas</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        {!selected ? (
          <>
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
                        {wp.name} {wp.found && <IonIcon icon={checkmarkCircle} color="success" />}
                        ({wp.lat.toFixed(2)}, {wp.lng.toFixed(2)})
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
            <IonCard>
              <IonCardHeader>
                <IonCardTitle>Mapa všech bodů</IonCardTitle>
              </IonCardHeader>
              <IonCardContent>
                <MapContainer position={selected} />
              </IonCardContent>
            </IonCard>
          </>
        ) : (
          <IonCard>
            <IonCardHeader color="primary">
              <IonCardTitle>
                {selected.name} 
                {selected.found && 
                  <IonBadge color="success" style={{ marginLeft: 10 }}>
                    <IonIcon icon={checkmarkCircle} /> Nalezeno
                  </IonBadge>
                }
              </IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <IonList lines="full">
                <IonItem>
                  <IonLabel>
                    <h2>Souřadnice</h2>
                    <p>{selected.lat.toFixed(6)}, {selected.lng.toFixed(6)}</p>
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
                
                {currentPos && !selected.found && (
                  <IonButton 
                    expand="block" 
                    color="success" 
                    onClick={handleOpenNameModal}
                    disabled={isSubmitting}
                    className="ion-margin-top"
                  >
                    <IonIcon icon={checkmarkCircle} slot="start" />
                    Označit jako nalezenou
                  </IonButton>
                )}
                
                <IonButton 
                  expand="block" 
                  color="medium" 
                  onClick={() => history.push('/waypoints')}
                  className="ion-margin-top"
                >
                  Změnit referenční bod
                </IonButton>
              </IonList>
            </IonCardContent>
          </IonCard>
        )}

        <IonModal isOpen={showNameModal} onDidDismiss={() => setShowNameModal(false)}>
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>Kdo našel kešku?</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <IonItem>
                <IonLabel position="stacked">Vaše jméno</IonLabel>
                <IonInput 
                  value={personName} 
                  onIonChange={e => setPersonName(e.detail.value?.toString() || '')}
                  placeholder="Zadejte své jméno"
                  required
                />
                <IonIcon icon={person} slot="start" />
              </IonItem>

              <div style={{ margin: '15px 0', textAlign: 'center' }}>
                <p>Vzdálenost od kešky: <b>{calculatedDistance.toFixed(2)} m</b></p>
              </div>
              
              <IonButton 
                expand="block" 
                color="success" 
                onClick={markAsFound}
                disabled={!personName.trim()}
                className="ion-margin-top"
              >
                <IonIcon icon={checkmarkCircle} slot="start" />
                Potvrdit nález
              </IonButton>
              <IonButton 
                expand="block" 
                color="medium" 
                onClick={() => setShowNameModal(false)} 
                className="ion-margin-top"
              >
                Zrušit
              </IonButton>
            </IonCardContent>
          </IonCard>
        </IonModal>

        <IonToast
          isOpen={showToast}
          onDidDismiss={() => setShowToast(false)}
          message={toastMessage}
          duration={3000}
          color={isSuccess ? 'success' : 'danger'}
          buttons={[
            {
              text: 'OK',
              role: 'cancel',
            }
          ]}
        />
      </IonContent>
    </IonPage>
  );
};

export default Navigate;