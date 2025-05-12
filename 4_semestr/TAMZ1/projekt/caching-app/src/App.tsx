import React from 'react';
import { Redirect, Route } from 'react-router-dom';
import { IonApp, IonRouterOutlet, setupIonicReact, IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonMenuButton, IonButtons, IonPage, IonMenuToggle, IonFooter } from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { NetworkProvider } from './context/NetworkContext';
import Home from './pages/Home';
import About from './pages/About';
import VersionContainer from './components/VersionContainer';
import Waypoints from './pages/Waypoints';
import Navigate from './pages/Navigate';
import NearestCache from './pages/NearestCache';
import CacheDetail from './pages/CacheDetail';
import AllCachesMap from './pages/AllCachesMap';
import Found from './pages/Found';
import OfflineIndicator from './components/OfflineIndicator';
import ImportExport from './pages/ImportExport';

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
  <NetworkProvider
    pingUrl={`${process.env.REACT_APP_API_URL}/api/health`}
    pingInterval={60000} // 1 minuta
  >
    <IonApp>
      <OfflineIndicator />
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
                <IonItem routerLink="/found-caches" routerDirection="root">Nalezeno</IonItem>
              </IonMenuToggle>
              <IonMenuToggle autoHide={false}>
                <IonItem routerLink="/nearest-cache" routerDirection="root">Najít nejbližší keš</IonItem>
              </IonMenuToggle>
              <IonMenuToggle autoHide={false}>
                <IonItem routerLink="/waypoints" routerDirection="root">Kešky</IonItem>
              </IonMenuToggle>
              <IonMenuToggle autoHide={false}>
                <IonItem routerLink="/about" routerDirection="root">O aplikaci</IonItem>
              </IonMenuToggle>
              <IonMenuToggle autoHide={false}>
                <IonItem routerLink="/importexport" routerDirection="root">Import / Export</IonItem>
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
              <IonTitle>GeoCaching</IonTitle>
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
            <Route exact path="/found-caches">
              <Found />
            </Route>
            <Route exact path="/about">
              <About />
            </Route>
            <Route exact path="/cache-detail">
              <CacheDetail />
            </Route>
            <Route exact path="/all-caches-map">
              <AllCachesMap />
            </Route>
            <Route exact path="/">
              <Redirect to="/home" />
            </Route>
            <Route exact path="/nearest-cache">
              <NearestCache />
            </Route>
            <Route exact path="/importexport">
              <ImportExport />
            </Route>
          </IonRouterOutlet>
          {/* <IonFooter>
            <VersionContainer />
          </IonFooter> */}
        </IonPage>
      </IonReactRouter>
    </IonApp>
  </NetworkProvider>
);

export default App;