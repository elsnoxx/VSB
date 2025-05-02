import React from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonList, IonItem, IonLabel, IonIcon } from '@ionic/react';
import WaypointsList from '../components/WaypointsList';

const Waypoints: React.FC = () => {
  return (
    <IonPage>
      <WaypointsList />
    </IonPage>
  );
};

export default Waypoints;