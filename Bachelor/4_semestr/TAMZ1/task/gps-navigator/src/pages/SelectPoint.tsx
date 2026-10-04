import React, { useRef, useEffect, useState } from 'react';
import { IonPage, IonHeader, IonToolbar, IonTitle, IonContent, IonButton, IonInput, IonItem, IonLabel, IonToast } from '@ionic/react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import { fromLonLat, toLonLat } from 'ol/proj';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import { Style, Icon } from 'ol/style';
import './SelectPoint.css';

const SelectPoint: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<Map | null>(null);
  const [selected, setSelected] = useState<{ lat: number; lng: number } | null>(null);
  const [name, setName] = useState('');
  const [toast, setToast] = useState<{ show: boolean; msg: string }>({ show: false, msg: '' });

  useEffect(() => {
    if (!mapRef.current) return;

    if (!mapObj.current) {
      mapObj.current = new Map({
        target: mapRef.current,
        layers: [
          new TileLayer({
            source: new OSM(),
          }),
        ],
        view: new View({
          center: fromLonLat([15, 50]),
          zoom: 6,
        }),
      });

      mapObj.current.on('singleclick', (evt) => {
        const [lng, lat] = toLonLat(evt.coordinate);
        setSelected({ lat, lng });

        // Přidej marker
        const vectorSource = new VectorSource();
        const marker = new Feature({
          geometry: new Point(evt.coordinate),
        });
        marker.setStyle(
          new Style({
            image: new Icon({
              src: 'https://openlayers.org/en/latest/examples/data/icon.png',
              anchor: [0.5, 1],
              scale: 0.7,
              color: '#007bff',
            }),
          })
        );
        vectorSource.addFeature(marker);

        // Odeber staré vektorové vrstvy
        mapObj.current?.getLayers().getArray()
          .filter(layer => layer instanceof VectorLayer)
          .forEach(layer => mapObj.current?.removeLayer(layer));

        // Přidej novou vektorovou vrstvu
        if (mapObj.current) {
          mapObj.current.addLayer(new VectorLayer({ source: vectorSource }));
        }
      });
    }
  }, []);

  useEffect(() => {
    const nav = localStorage.getItem('navigateTo');
    if (nav) {
      const navPoint = JSON.parse(nav);
      setSelected(navPoint);
      localStorage.removeItem('navigateTo');
    }
  }, []);

  const handleSave = () => {
    if (!selected || !name.trim()) {
      setToast({ show: true, msg: 'Vyberte bod na mapě a zadejte název!' });
      return;
    }
    const waypoints = JSON.parse(localStorage.getItem('waypoints') || '[]');
    waypoints.push({
      name,
      date: new Date().toISOString(),
      lat: selected.lat,
      lng: selected.lng,
    });
    localStorage.setItem('waypoints', JSON.stringify(waypoints));
    setToast({ show: true, msg: 'Bod uložen!' });
    setName('');
    setSelected(null);
  };

  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Vybrat bod na mapě</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent>
        <div ref={mapRef} style={{ width: '100%', height: 300, margin: '16px 0' }} />
        <IonItem>
          <IonLabel position="stacked">Název bodu</IonLabel>
          <IonInput value={name} onIonChange={e => setName(e.detail.value!)} placeholder="Zadejte název" />
        </IonItem>
        <IonItem>
          <IonLabel>Pozice</IonLabel>
          <IonLabel slot="end">
            {selected ? `${selected.lat.toFixed(5)}, ${selected.lng.toFixed(5)}` : 'Není vybráno'}
          </IonLabel>
        </IonItem>
        <IonButton expand="block" onClick={handleSave} style={{ margin: 16 }}>
          Uložit bod
        </IonButton>
        <IonToast
          isOpen={toast.show}
          message={toast.msg}
          duration={1500}
          onDidDismiss={() => setToast({ show: false, msg: '' })}
        />
        <div style={{ fontSize: '0.8em', textAlign: 'right', color: '#888', marginTop: 8 }}>
          © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors
        </div>
      </IonContent>
    </IonPage>
  );
};

export default SelectPoint;