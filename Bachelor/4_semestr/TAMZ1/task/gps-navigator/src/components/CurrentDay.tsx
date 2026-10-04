import React, { useEffect, useState } from 'react';
import { IonCard, IonCardHeader, IonCardTitle, IonCardContent } from '@ionic/react';

const CurrentDay: React.FC = () => {
    return(
        <IonCard>
            <IonCardHeader>
                <IonCardTitle>{new Date().toLocaleDateString()} | {new Date().toLocaleTimeString()}</IonCardTitle>
            </IonCardHeader>
            <IonCardContent>
            <div style={{textAlign: 'center' }}>
                Aplikace pro trekování aktuální polohy a zobrazení mapy
            </div>
            </IonCardContent>
        </IonCard>
    );
}

export default CurrentDay;