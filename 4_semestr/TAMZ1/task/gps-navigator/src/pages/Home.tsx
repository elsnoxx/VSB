import React, { useEffect, useState } from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar } from '@ionic/react';
import { Geolocation } from '@capacitor/geolocation';
import CurrentPosition from '../components/CurrentPosition';
import MapContainer from '../components/MapContainer';
import CurrentDay from '../components/CurrentDay';
import SaveWaypoints from '../components/SaveWaypoints';
import './Home.css';

const Home: React.FC = () => {
  const [position, setPosition] = useState<{lat: number, lng: number} | null>(null);

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

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Current Position</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonHeader collapse="condense">
          <IonToolbar>
            <IonTitle size="large">Current Position</IonTitle>
          </IonToolbar>
        </IonHeader>
        <CurrentDay />
        <MapContainer position={position} />
        <CurrentPosition />
        <SaveWaypoints position={position} />
      </IonContent>
    </IonPage>
  );
};

export default Home;