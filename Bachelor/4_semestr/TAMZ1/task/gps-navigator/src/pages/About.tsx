import React from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonList, IonItem, IonLabel, IonIcon } from '@ionic/react';
import { calendar, mailOutline, person } from 'ionicons/icons';
import './About.css';

const About: React.FC = () => {
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>O aplikaci</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonCard>
          <IonCardHeader>
            <IonCardTitle>Aplikace pro geolokaci</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <p>Tato aplikace slouží k zobrazení aktuální polohy uživatele na mapě a využívá geolokační služby pro získání přesných souřadnic.</p>
            <IonList>
              <IonItem>
              <IonIcon icon={person} slot="start" />
                <IonLabel>Autor</IonLabel>
                <IonLabel slot="end">Richard Ficek</IonLabel>
              </IonItem>
              <IonItem>
              <IonIcon icon={calendar} slot="start" />
                <IonLabel>Datum</IonLabel>
                <IonLabel slot="end">2023-10-01</IonLabel>
              </IonItem>
              <IonItem href="mailto:richard.ficek.st@vsb.cz" target="_blank">
                <IonIcon icon={mailOutline} slot="start" />
                <IonLabel>Email</IonLabel>
                <IonLabel slot="end" color="primary">richard.ficek.st@vsb.cz</IonLabel>
              </IonItem>
            </IonList>
            <p style={{marginTop: 16}}>V případě dotazů nebo problémů mě neváhejte kontaktovat.</p>
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default About;