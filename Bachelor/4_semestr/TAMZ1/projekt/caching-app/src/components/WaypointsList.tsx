import React from 'react';
import { IonList, IonItem, IonLabel } from '@ionic/react';

type Waypoint = {
  name: string;
  date: string;
  lat: number;
  lng: number;
  accuracy?: number;
  altitude?: number | null;
  altitudeAccuracy?: number | null;
};

interface WaypointsListProps {
  waypoints: Waypoint[];
  onSelect: (wp: Waypoint) => void;
}

const WaypointsList: React.FC<WaypointsListProps> = ({ waypoints, onSelect }) => (
  <IonList>
    {waypoints.map((wp, idx) => (
      <IonItem button key={idx} onClick={() => onSelect(wp)}>
        <IonLabel>
          <h2>{wp.name}</h2>
          <p>{wp.date} | {wp.lat.toFixed(2)}, {wp.lng.toFixed(2)}</p>
        </IonLabel>
      </IonItem>
    ))}
  </IonList>
);

export default WaypointsList;