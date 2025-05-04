import React, { useEffect, useState } from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonButton, IonCard, IonCardHeader, IonCardTitle, IonCardContent } from '@ionic/react';
import { Geolocation } from '@capacitor/geolocation';
import CurrentDay from '../components/CurrentDay';
import './Home.css';
import { useHistory } from 'react-router-dom';

const Home: React.FC = () => {
  const [position, setPosition] = useState<{lat: number, lng: number} | null>(null);
  const history = useHistory();

  useEffect(() => {
    const getCurrentPosition = async () => {
      const coordinates = await Geolocation.getCurrentPosition();
      setPosition({
        lat: coordinates.coords.latitude,
        lng: coordinates.coords.longitude
      });
    };
    getCurrentPosition();
  }, []);

  const handleShowAllCaches = () => {
    history.push('/waypoints');
  };

  const handleShowNearestCache = () => {
    history.push('/nearest-cache');
  };

  return (
    <IonPage>
      <IonHeader translucent={true}>
        <IonToolbar color="primary">
          <IonTitle>GPS Navigator</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonHeader collapse="condense">
          <IonToolbar>
            <IonTitle size="large">Hlavní plocha</IonTitle>
          </IonToolbar>
        </IonHeader>
        <CurrentDay />
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>Vítejte v aplikaci pro hledání cachek!</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <p>Zde můžete:</p>
            <ul>
              <li>Získat informace o své aktuální poloze</li>
              <li>Zobrazit všechny dostupné cache</li>
              <li>Najít nejbližší cache</li>
            </ul>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16 }}>
              <IonButton onClick={handleShowAllCaches} color="secondary">
                Zobrazit všechny cache
              </IonButton>
              <IonButton onClick={handleShowNearestCache} color="tertiary">
                Najít nejbližší cache
              </IonButton>
            </div>
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default Home;