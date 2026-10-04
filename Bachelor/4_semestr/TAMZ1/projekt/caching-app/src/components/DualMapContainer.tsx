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
import './DualMapContainer.css';

type Props = {
  current: { lat: number, lng: number } | null;
  target: { lat: number, lng: number } | null;
};

const DualMapContainer: React.FC<Props> = ({ current, target }) => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<Map | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    // Inicializace mapy pouze jednou
    if (!mapObj.current) {
      mapObj.current = new Map({
        target: mapRef.current,
        controls: defaultControls({ attribution: false }),
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
    }

    // Připrav marker(y)
    const features: Feature[] = [];
    if (current) {
      const marker = new Feature({
        geometry: new Point(fromLonLat([current.lng, current.lat])),
        name: 'Aktuální pozice',
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
      features.push(marker);
    }
    if (target) {
      const marker = new Feature({
        geometry: new Point(fromLonLat([target.lng, target.lat])),
        name: 'Referenční bod',
      });
      marker.setStyle(
        new Style({
          image: new Icon({
            src: 'https://openlayers.org/en/latest/examples/data/icon.png',
            anchor: [0.5, 1],
            scale: 0.7,
            color: '#ff0000',
          }),
        })
      );
      features.push(marker);
    }

    // Odeber staré vektorové vrstvy (pokud existují)
    mapObj.current.getLayers().getArray()
      .filter(layer => layer instanceof VectorLayer)
      .forEach(layer => mapObj.current?.removeLayer(layer));

    // Přidej novou vektorovou vrstvu
    if (features.length > 0) {
      const vectorSource = new VectorSource({ features });
      const vectorLayer = new VectorLayer({ source: vectorSource });
      mapObj.current.addLayer(vectorLayer);

      // Pokud je aktuální pozice, centrovat na ni a přiblížit
      if (current) {
        const coord = fromLonLat([current.lng, current.lat]);
        mapObj.current.getView().setCenter(coord);
        mapObj.current.getView().setZoom(15);
      } else if (target) {
        // Pokud je jen target
        const coord = fromLonLat([target.lng, target.lat]);
        mapObj.current.getView().setCenter(coord);
        mapObj.current.getView().setZoom(15);
      }
    }
  }, [current, target]);

  return (
    <>
      <div ref={mapRef} style={{ width: '100%', height: 300, marginBottom: 0 }} />
      <div style={{ fontSize: '0.8em', textAlign: 'right', color: '#888', marginTop: 8 }}>
        © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors
      </div>
    </>
  );
};

export default DualMapContainer;