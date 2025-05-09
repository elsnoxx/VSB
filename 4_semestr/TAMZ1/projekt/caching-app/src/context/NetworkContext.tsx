import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { IonToast } from '@ionic/react';

interface NetworkContextType {
  isOnline: boolean;
  lastChecked: Date | null;
}

const NetworkContext = createContext<NetworkContextType>({
  isOnline: true,
  lastChecked: null
});

export const useNetwork = () => useContext(NetworkContext);

interface NetworkProviderProps {
  children: ReactNode;
  pingUrl?: string;
  pingInterval?: number;
}

export const NetworkProvider: React.FC<NetworkProviderProps> = ({
  children,
  pingUrl = 'https://www.google.com',
  pingInterval = 30000
}) => {
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);
  const [showToast, setShowToast] = useState<boolean>(false);

  const checkServerConnection = async (url: string) => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(url, {
        method: 'GET',
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.log('Kontrola připojení selhala:', error);
      return false;
    }
  };

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

    const intervalCheck = setInterval(async () => {
      const serverReachable = await checkServerConnection(pingUrl);

      if (serverReachable !== isOnline) {
        setIsOnline(serverReachable);
        setShowToast(!serverReachable);
      }
    }, pingInterval);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      clearInterval(intervalCheck);
    };
  }, [isOnline, pingUrl, pingInterval]);

  return (
    <NetworkContext.Provider value={{ isOnline, lastChecked: null }}>
      {children}

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
            handler: () => setShowToast(false),
          },
        ]}
      />
    </NetworkContext.Provider>
  );
};