import React, { useEffect, useRef } from 'react';
import { useHistory } from 'react-router-dom';
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
import { defaults as defaultControls } from 'ol/control';
import Overlay from 'ol/Overlay';
import './MapContainerMultiple.css';

type Cache = { 
  lat: number; 
  lng: number; 
  name?: string;
  found?: boolean; 
};

type Props = {
  caches: Cache[];
  onMarkerClick?: (cache: Cache) => void;
  myPosition?: { lat: number; lng: number } | null;
  foundCacheNames?: string[];
};

const MapContainerMultiple: React.FC<Props> = ({ caches, onMarkerClick, myPosition, foundCacheNames = [] }) => {
  const history = useHistory();
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const mapObj = useRef<Map | null>(null);
  const overlayRef = useRef<Overlay | null>(null);

  // Inicializace mapy a overlaye
  useEffect(() => {
    if (!mapRef.current) return;

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

    // Overlay inicializuj až když je popupRef k dispozici
    if (popupRef.current && !overlayRef.current && mapObj.current) {
      overlayRef.current = new Overlay({
        element: popupRef.current,
        autoPan: true,
      });
      mapObj.current.addOverlay(overlayRef.current);

      mapObj.current.on('singleclick', function (evt) {
        if (!mapObj.current || !overlayRef.current || !popupRef.current) return;
        
        // Najdi feature na kliknutí
        const feature = mapObj.current.forEachFeatureAtPixel(evt.pixel, f => f as Feature<Point>);
        
        if (feature) {
          const name = feature.get('name') || 'Keška';
          const isUserLocation = feature.get('isUserLocation') === true;
          
          // Pokud to není uživatelská poloha, zpracuj jako kešku
          if (!isUserLocation) {
            overlayRef.current.setPosition(evt.coordinate);
            
            // Zjisti, zda je keška nalezena
            const isFound = feature.get('found') === true || foundCacheNames?.includes(name);
            
            // Zobraz tooltip s informací
            const foundBadge = isFound ? 
              '<span style="background:#28ba62; color:white; padding:2px 6px; border-radius:10px; margin-left:6px; font-size:10px;">✓ Nalezeno</span>' : 
              '';
              
            popupRef.current.innerHTML = `
              <div style="padding:8px; color:black">
                <b>${name}</b> ${foundBadge}
                <div style="font-size:12px; margin-top:4px;">
                  <a href="#" id="cache-detail-link" style="color:#3880ff; text-decoration:underline; cursor:pointer;">
                    Zobrazit detail
                  </a>
                </div>
              </div>
            `;

            // Přidat event listener na odkaz
            const linkElement = popupRef.current.querySelector('#cache-detail-link');
            if (linkElement) {
              linkElement.addEventListener('click', (e) => {
                e.preventDefault(); // Zabrání reloadu stránky
                let cache = caches.find(c => c.name === name);
                if (!cache) {
                  const geometry = feature.getGeometry() as Point;
                  const coords = geometry ? geometry.getCoordinates() : [0, 0];
                  const [lng, lat] = toLonLat(coords);
                  cache = {
                    name,
                    lat,
                    lng,
                    found: isFound
                  };
                }
                localStorage.setItem('selectedCache', JSON.stringify({
                  ...cache,
                  found: isFound
                }));
                history.push('/cache-detail');
              });
            }
            
            // Najdi kešku v poli a předej ji nadřazené komponentě
            if (onMarkerClick) {
              const cache = caches.find(c => c.name === name);
              if (cache) {
                // Označit cache jako nalezenou, pokud je v seznamu nalezených
                const isFound = foundCacheNames?.includes(name);
                onMarkerClick({...cache, found: isFound});
              }
            }
          } else {
            // Je to uživatelská poloha, zobraz jiný obsah v tooltiipu
            overlayRef.current.setPosition(evt.coordinate);
            popupRef.current.innerHTML = `<div style="padding:8px; color:black"><b>Moje poloha</b></div>`;
          }
        } else {
          // Klikli jsme mimo markery, skryj popup
          overlayRef.current.setPosition(undefined);
        }
      });
    }
  }, [onMarkerClick, foundCacheNames]);

  // Aktualizace markerů při změně caches, myPosition a foundCacheNames
  useEffect(() => {
    if (!mapObj.current) return;

    // Odeber staré vektorové vrstvy
    mapObj.current.getLayers().getArray()
      .filter(layer => layer instanceof VectorLayer)
      .forEach(layer => mapObj.current?.removeLayer(layer));

    // Přidej nové markery
    const features = caches.map(cache => {
      const isFound = cache.found || foundCacheNames?.includes(cache.name || '');
      
      const feature = new Feature({
        geometry: new Point(fromLonLat([cache.lng, cache.lat])),
        name: cache.name || 'Keška',
        found: isFound
      });
      
      feature.setStyle(
        new Style({
          image: new Icon({
            // Použij jiný obrázek pro nalezené kešky
            src: isFound 
              ? 'https://maps.google.com/mapfiles/ms/icons/green-dot.png' 
              : 'https://maps.google.com/mapfiles/ms/icons/red-dot.png',
            anchor: [0.5, 1],
            scale: 0.7,
          }),
        })
      );
      return feature;
    });

    // Přidej marker pro uživatele (modrý)
    if (myPosition) {
      const userFeature = new Feature({
        geometry: new Point(fromLonLat([myPosition.lng, myPosition.lat])),
        name: 'Moje poloha',
        isUserLocation: true
      });
      userFeature.setStyle(
        new Style({
          image: new Icon({
            src: 'https://maps.google.com/mapfiles/ms/icons/blue-dot.png',
            anchor: [0.5, 1],
            scale: 1,
          }),
        })
      );
      features.push(userFeature);
    }

    const vectorSource = new VectorSource({ features });
    const vectorLayer = new VectorLayer({ source: vectorSource });
    mapObj.current.addLayer(vectorLayer);

    if (features.length > 0) {
      const extent = vectorSource.getExtent();
      mapObj.current.getView().fit(extent, { maxZoom: 15, duration: 500 });
    }
  }, [caches, myPosition, foundCacheNames]);

  return (
    <>
      <div ref={mapRef} style={{ width: '100%', height: '80%', marginBottom: 16 }} />
      <div ref={popupRef} className="ol-popup" />
      <div style={{ fontSize: '0.8em', textAlign: 'right', color: '#888' }}>
        © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors
      </div>
    </>
  );
};

export default MapContainerMultiple;