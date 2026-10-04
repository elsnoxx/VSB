import React, { useState } from "react";
import { IonButton, IonModal, IonHeader, IonToolbar, IonTitle, IonContent, IonInput, IonItem, IonLabel, IonFooter, IonToast } from '@ionic/react';
import { useHistory } from 'react-router-dom';

type WaypointProps = {
    position: {
        lat: number;
        lng: number;
        accuracy?: number;
        altitude?: number | null;
        altitudeAccuracy?: number | null;
    } | null;
};

const SaveWaypoints: React.FC<WaypointProps> = ({ position }) => {
    const [showModal, setShowModal] = useState(false);
    const [waypointName, setWaypointName] = useState('');
    const [toastMessage, setToastMessage] = useState('');
    const [showToast, setShowToast] = useState(false);
    const history = useHistory();

    const handleSave = () => {
        if (!waypointName.trim()) {
            setToastMessage('Zadejte název bodu!');
            setShowToast(true);
            return;
        }
        if (!position) {
            setToastMessage('Pozice není k dispozici!');
            setShowToast(true);
            return;
        }
        const waypoints = JSON.parse(localStorage.getItem('waypoints') || '[]');
        waypoints.push({
            name: waypointName,
            date: new Date().toISOString(),
            lat: position.lat,
            lng: position.lng,
            accuracy: position.accuracy,
            altitude: position.altitude,
            altitudeAccuracy: position.altitudeAccuracy,
        });
        localStorage.setItem('waypoints', JSON.stringify(waypoints));
        setToastMessage('Waypoint uložen!');
        setShowToast(true);
        setWaypointName('');
        setTimeout(() => {
            setShowModal(false);
        }, 1000);
    };

    return (
        <>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <IonButton expand="block" color="primary" onClick={() => setShowModal(true)}>
                    Přidat waypoint
                </IonButton>
                <IonButton expand="block" color="secondary" onClick={() => history.push('/select-point')}>
                    Vybrat bod na mapě
                </IonButton>
            </div>
            <IonModal isOpen={showModal} onDidDismiss={() => setShowModal(false)}>
                <IonHeader>
                    <IonToolbar>
                        <IonTitle>Přidat waypoint</IonTitle>
                    </IonToolbar>
                </IonHeader>
                <IonContent>
                    <IonItem>
                        <IonLabel position="stacked">Název bodu</IonLabel>
                        <IonInput
                            value={waypointName}
                            placeholder="Zadejte název"
                            onIonChange={e => setWaypointName(e.detail.value!)}
                        />
                    </IonItem>
                    <IonItem>
                        <IonLabel>Poloha (lat, lng)</IonLabel>
                        <IonLabel slot="end">
                            {position ? `${position.lat.toFixed(2)}, ${position.lng.toFixed(2)}` : 'N/A'}
                        </IonLabel>
                    </IonItem>
                    <IonItem>
                        <IonLabel>Přesnost</IonLabel>
                        <IonLabel slot="end">
                            {position?.accuracy ? `${position.accuracy} m` : 'N/A'}
                        </IonLabel>
                    </IonItem>
                    <IonItem>
                        <IonLabel>Nadmořská výška</IonLabel>
                        <IonLabel slot="end">
                            {position?.altitude !== undefined && position?.altitude !== null ? `${position.altitude} m` : 'N/A'}
                        </IonLabel>
                    </IonItem>
                    <IonItem>
                        <IonLabel>Přesnost výšky</IonLabel>
                        <IonLabel slot="end">
                            {position?.altitudeAccuracy !== undefined && position?.altitudeAccuracy !== null ? `${position.altitudeAccuracy} m` : 'N/A'}
                        </IonLabel>
                    </IonItem>
                </IonContent>
                <IonFooter>
                    <IonToolbar>
                        <IonButton expand="block" onClick={handleSave}>Uložit</IonButton>
                        <IonButton expand="block" color="medium" onClick={() => setShowModal(false)}>Zavřít</IonButton>
                    </IonToolbar>
                </IonFooter>
            </IonModal>
            <IonToast
                isOpen={showToast}
                onDidDismiss={() => setShowToast(false)}
                message={toastMessage}
                duration={1500}
                position="top"
                color="success"
            />
        </>
    );
};

export default SaveWaypoints;