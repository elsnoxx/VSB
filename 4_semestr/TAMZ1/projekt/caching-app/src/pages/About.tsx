import React from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonCard, IonCardHeader, IonCardTitle, IonCardContent, IonList, IonItem, IonLabel, IonIcon } from '@ionic/react';
import { calendar, mailOutline, person, mapOutline } from 'ionicons/icons';
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
            <IonCardTitle>Caching aplikace</IonCardTitle>
          </IonCardHeader>
          <IonCardContent>
            <p>
              Tato aplikace slouží pro vyhledávání, správu a navigaci ke geocachingovým schránkám (cachek).
              Umožňuje zobrazit seznam dostupných cachek, najít nejbližší cache podle aktuální polohy, zobrazit detail cache včetně mapy a navigovat k vybrané cache.
            </p>
            <IonList>
              <IonItem>
                <IonIcon icon={person} slot="start" />
                <IonLabel>Autor</IonLabel>
                <IonLabel slot="end">Richard Ficek</IonLabel>
              </IonItem>
              <IonItem>
                <IonIcon icon={calendar} slot="start" />
                <IonLabel>Datum</IonLabel>
                <IonLabel slot="end">2025-05-05</IonLabel>
              </IonItem>
              <IonItem>
                <IonIcon icon={mapOutline} slot="start" />
                <IonLabel>Verze</IonLabel>
                <IonLabel slot="end">1.9</IonLabel>
              </IonItem>
              <IonItem href="mailto:richard.ficek.st@vsb.cz" target="_blank">
                <IonIcon icon={mailOutline} slot="start" />
                <IonLabel>Email</IonLabel>
                <IonLabel slot="end" color="primary">richard.ficek.st@vsb.cz</IonLabel>
              </IonItem>
            </IonList>
            <p style={{marginTop: 16}}>
              V případě dotazů nebo problémů mě neváhejte kontaktovat.<br />
              <b>Funkce aplikace:</b>
              <ul>
                <li>Zobrazení seznamu cachek a jejich detailů</li>
                <li>Vyhledání nejbližší cache podle polohy</li>
                <li>Zobrazení cache na mapě</li>
                <li>Navigace ke zvolené cache</li>
                <li>Ukládání vlastních bodů zájmu</li>
              </ul>
            </p>
          </IonCardContent>
        </IonCard>
      </IonContent>
    </IonPage>
  );
};

export default About;