import React, { useEffect, useState } from 'react';
import { IonCard, IonCardHeader, IonCardTitle, IonCardContent } from '@ionic/react';


const VersionContainer: React.FC = () => {
    return(
        <IonCard>
            <IonCardContent>
            <div style={{textAlign: 'center' }}>
                Verze aplikace: {process.env.REACT_APP_VERSION}
            </div>
            </IonCardContent>
        </IonCard>
    );
}

export default VersionContainer;