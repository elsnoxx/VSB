import React, { useEffect, useRef } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import { fromLonLat } from 'ol/proj';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import { Style, Icon } from 'ol/style';
import { defaults as defaultControls } from 'ol/control';
import './MapContainer.css';

type Props = {
  positions: { lat: number, lng: number, name?: string }[];
};

const MapContainer: React.FC<Props> = ({ positions }) => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<Map | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Inicializace mapy pouze jednou
    if (!mapObj.current) {
      mapObj.current = new Map({
        target: mapRef.current,
        controls: defaultControls({ attribution: false }), // vypne attribution control
        layers: [
          new TileLayer({
            source: new OSM(),
          }),
        ],
        view: new View({
          center: fromLonLat([15, 50]), // výchozí střed Evropy
          zoom: 6,
        }),
      });
    }

    // Pokud jsou pozice, přidej markery a posuň mapu
    if (positions && mapObj.current) {
      const markers = positions.map(({ lat, lng }) => {
        const marker = new Feature({
          geometry: new Point(fromLonLat([lng, lat])),
        });
        marker.setStyle(
          new Style({
            image: new Icon({
              src: 'https://openlayers.org/en/latest/examples/data/icon.png',
              anchor: [0.5, 1],
              scale: 0.7,
            }),
          })
        );
        return marker;
      });

      const vectorSource = new VectorSource({
        features: markers,
      });
      const vectorLayer = new VectorLayer({
        source: vectorSource,
      });

      // Odeber staré vektorové vrstvy (pokud existují)
      mapObj.current.getLayers().getArray()
        .filter(layer => layer instanceof VectorLayer)
        .forEach(layer => mapObj.current?.removeLayer(layer));

      mapObj.current.addLayer(vectorLayer);
      if (positions.length > 0) {
        const { lat, lng } = positions[0];
        mapObj.current.getView().setCenter(fromLonLat([lng, lat]));
        mapObj.current.getView().setZoom(15);
      }
    }
  }, [positions]);

  return (
    <>
      <div ref={mapRef} style={{ width: '100%', height: 300, marginBottom: 16 }} />
      <div style={{ fontSize: '0.8em', textAlign: 'right', color: '#888' }}>
        © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors
      </div>
    </>
  );
};

export default MapContainer;