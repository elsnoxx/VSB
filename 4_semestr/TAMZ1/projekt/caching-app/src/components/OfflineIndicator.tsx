import React, { useEffect, useState } from 'react';
import { IonToast } from '@ionic/react';

const OfflineIndicator: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);
  const [showToast, setShowToast] = useState<boolean>(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowToast(false);
    };
    const handleOffline = () => {
      setIsOnline(false);
      setShowToast(true);
    };

    if (!navigator.onLine) {
      setShowToast(true);
    }

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <IonToast
      isOpen={showToast}
      onDidDismiss={() => setShowToast(false)}
      message="Jste offline. Některé funkce mohou být omezeny."
      position="top"
      color="warning"
      duration={3000}
      buttons={[
        {
          text: 'OK',
          role: 'cancel',
          handler: () => setShowToast(false)
        }
      ]}
    />
  );
};

export default OfflineIndicator;