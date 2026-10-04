import React from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonList, IonItem, IonLabel, IonIcon } from '@ionic/react';
import { calendar, mailOutline, person, mapOutline } from 'ionicons/icons';
import CsvImportExport from '../components/CsvImportExport';

const ImportExport: React.FC = () => {
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Import / Export</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>CSV Import/Export</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <CsvImportExport />
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default ImportExport;