import React from 'react';
import { Redirect, Route } from 'react-router-dom';
import { IonApp, IonRouterOutlet, setupIonicReact, IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonMenuButton, IonButtons, IonPage, IonMenuToggle, IonFooter } from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import Home from './pages/Home';
import About from './pages/About';
import VersionContainer from './components/VersionContainer';
import Waypoints from './pages/Waypoints';
import Navigate from './pages/Navigate';
import NearestCache from './pages/NearestCache';
import CacheDetail from './pages/CacheDetail';

import '@ionic/react/css/core.css';
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';
import './theme/variables.css';

setupIonicReact();

const App: React.FC = () => (
  <IonApp>
    <IonReactRouter>
      {/* Menu */}
      <IonMenu contentId="main-content">
        <IonHeader>
          <IonToolbar>
            <IonTitle>Menu</IonTitle>
          </IonToolbar>
        </IonHeader>
        <IonContent>
          <IonList>
            <IonMenuToggle autoHide={false}>
              <IonItem routerLink="/home" routerDirection="root">Domů</IonItem>
            </IonMenuToggle>
            <IonMenuToggle autoHide={false}>
              <IonItem routerLink="/navigate" routerDirection="root">Navigovat</IonItem>
            </IonMenuToggle>
            <IonMenuToggle autoHide={false}>
              <IonItem routerLink="/waypoints" routerDirection="root">Kešky</IonItem>
            </IonMenuToggle>
            <IonMenuToggle autoHide={false}>
              <IonItem routerLink="/about" routerDirection="root">O aplikaci</IonItem>
            </IonMenuToggle>
            {/* Další odkazy */}
          </IonList>
        </IonContent>
      </IonMenu>
      {/* Hlavní obsah */}
      <IonPage id="main-content">
        <IonHeader>
          <IonToolbar>
            <IonButtons slot="start">
              <IonMenuButton />
            </IonButtons>
            <IonTitle>GPS Navigator</IonTitle>
          </IonToolbar>
        </IonHeader>
        <IonRouterOutlet>
          <Route exact path="/home">
            <Home />
          </Route>
          <Route exact path="/navigate">
            <Navigate />
          </Route>
          <Route exact path="/waypoints">
            <Waypoints />
          </Route>
          <Route exact path="/about">
            <About />
          </Route>
          <Route exact path="/cache-detail">
            <CacheDetail />
          </Route>
          <Route exact path="/">
            <Redirect to="/home" />
          </Route>
          <Route exact path="/nearest-cache">
            <NearestCache />
          </Route>
        </IonRouterOutlet>
        {/* <IonFooter>
          <VersionContainer />
        </IonFooter> */}
      </IonPage>
    </IonReactRouter>
  </IonApp>
);

export default App;