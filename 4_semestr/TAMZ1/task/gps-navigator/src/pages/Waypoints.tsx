import React, { useState } from 'react';
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
  IonButton
} from '@ionic/react';
import { locationOutline } from 'ionicons/icons';
import { useHistory } from 'react-router-dom';
import WaypointsList from '../components/WaypointsList';

type Waypoint = {
  name: string;
  date: string;
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
};

const Waypoints: React.FC = () => {
  const history = useHistory();

  const handleSelect = (wp: Waypoint) => {
    // Ulož vybraný bod do localStorage (nebo jinam podle logiky aplikace)
    localStorage.setItem('navigateTo', JSON.stringify(wp));
    // Přesměruj na stránku navigace
    history.push('/navigate');
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Uložené body</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>
              <IonIcon icon={locationOutline} style={{ marginRight: 8, verticalAlign: 'middle' }} />
              Seznam waypointů
            </IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <WaypointsList onSelect={handleSelect} />
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default Waypoints;