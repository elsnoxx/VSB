import React, { useEffect, useState } from 'react';
import { 
  IonContent, 
  IonHeader, 
  IonPage, 
  IonTitle, 
  IonToolbar, 
  IonButton, 
  IonCard, 
  IonCardHeader, 
  IonCardTitle, 
  IonCardContent,
  IonGrid,
  IonRow,
  IonCol,
  IonIcon,
  IonItem,
  IonList,
  IonLabel,
  IonChip,
  IonAvatar,
  IonText,
  IonModal,
  IonInput,
  IonAlert
} from '@ionic/react';
import { Geolocation } from '@capacitor/geolocation';
import { mapOutline, navigateCircleOutline, locationOutline, compass, person } from 'ionicons/icons';
import CurrentDay from '../components/CurrentDay';
import './Home.css';
import { useHistory } from 'react-router-dom';

const Home: React.FC = () => {
  const [position, setPosition] = useState<{lat: number, lng: number} | null>(null);
  const [showNameModal, setShowNameModal] = useState(false);
  const [personName, setPersonName] = useState('');
  const history = useHistory();

  // Kontrola existence jména při načtení stránky
  useEffect(() => {
    const getCurrentPosition = async () => {
      const coordinates = await Geolocation.getCurrentPosition();
      setPosition({
        lat: coordinates.coords.latitude,
        lng: coordinates.coords.longitude
      });
    };
    getCurrentPosition();

    // Kontrola, zda existuje personName v localStorage
    const savedName = localStorage.getItem('personName');
    if (!savedName) {
      // Pokud neexistuje, zobraz modál pro zadání jména
      setShowNameModal(true);
    } else {
      setPersonName(savedName);
    }
  }, []);

  const handleShowAllCaches = () => {
    history.push('/waypoints');
  };

  const handleShowNearestCache = () => {
    history.push('/nearest-cache');
  };
  
  const handleNavigate = () => {
    history.push('/navigate');
  };

  // Uložení jména uživatele
  const savePersonName = () => {
    if (personName.trim()) {
      localStorage.setItem('personName', personName.trim());
      setShowNameModal(false);
    }
  };

  return (
    <IonPage>
      <IonHeader translucent>
        <IonToolbar color="primary">
          <IonTitle>GeoCaching</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen className="ion-padding">
        {/* Uživatelská identifikace */}
        {personName && (
          <IonGrid className="user-chip-bar">
            <IonRow>
              <IonCol size="12">
                <IonChip color="primary" className="user-chip" style={{ width: '100%' }}>
                  <IonAvatar>
                    <IonIcon icon={person} />
                  </IonAvatar>
                  <IonLabel>Přihlášen jako: {personName}</IonLabel>
                </IonChip>
              </IonCol>
            </IonRow>
          </IonGrid>
        )}

        <CurrentDay />        

        {/* Úvodní karta */}
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>Vítejte v aplikaci pro hledání keší!</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <IonText color="medium">
              <p>Prozkoumávejte svět, hledejte poklady a bavte se s GPS Navigator</p>
            </IonText>
            <IonList lines="none">
              <IonItem>
                <IonIcon icon={locationOutline} slot="start" color="primary" />
                <IonLabel>Získat informace o své aktuální poloze</IonLabel>
              </IonItem>
              <IonItem>
                <IonIcon icon={mapOutline} slot="start" color="secondary" />
                <IonLabel>Zobrazit všechny dostupné keše</IonLabel>
              </IonItem>
              <IonItem>
                <IonIcon icon={navigateCircleOutline} slot="start" color="tertiary" />
                <IonLabel>Najít nejbližší keš</IonLabel>
              </IonItem>
            </IonList>
          </IonCardContent>
        </IonCard>

        {/* Akční tlačítka */}
        <IonGrid>
          <IonRow>
            <IonCol>
              <IonButton
                expand="block"
                color="secondary"
                onClick={handleShowAllCaches}
                className="ion-margin-bottom"
              >
                <IonIcon icon={mapOutline} slot="start" />
                Všechny keše
              </IonButton>
            </IonCol>
            <IonCol>
              <IonButton
                expand="block"
                color="tertiary"
                onClick={handleShowNearestCache}
                className="ion-margin-bottom"
              >
                <IonIcon icon={navigateCircleOutline} slot="start" />
                Nejbližší keš
              </IonButton>
            </IonCol>
            <IonCol>
              <IonButton
                expand="block"
                color="primary"
                onClick={handleNavigate}
              >
                <IonIcon icon={compass} slot="start" />
                Navigace
              </IonButton>
            </IonCol>
          </IonRow>
        </IonGrid>

        {/* Modální okno pro zadání jména */}
        <IonModal isOpen={showNameModal} backdropDismiss={false}>
          <IonCard>
            <IonCardHeader>
              <IonCardTitle>Zadejte své jméno</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
              <p>Pro plné využití aplikace je třeba zadat vaše jméno.</p>
              <p>Toto jméno bude použito při označení kešky jako nalezené.</p>
              <IonItem>
                <IonLabel position="floating">Jméno hledače</IonLabel>
                <IonInput
                  value={personName}
                  onIonChange={e => setPersonName(e.detail.value?.toString() || '')}
                  placeholder="Např. Jan Novák"
                />
                <IonIcon icon={person} slot="start" />
              </IonItem>
              <IonButton
                expand="block"
                color="primary"
                onClick={savePersonName}
                disabled={!personName.trim()}
                className="ion-margin-top"
              >
                Potvrdit
              </IonButton>
            </IonCardContent>
          </IonCard>
        </IonModal>
      </IonContent>
    </IonPage>
  );
};

export default Home;