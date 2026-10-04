import React, { useRef } from 'react';
import { IonButton, IonIcon } from '@ionic/react';
import { cloudDownloadOutline, cloudUploadOutline } from 'ionicons/icons';

const CsvImportExport: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Export foundCaches do CSV (pouze jména kešek)
  const handleExport = () => {
    const foundCaches = JSON.parse(localStorage.getItem('foundCaches') || '[]');
    if (!Array.isArray(foundCaches) || foundCaches.length === 0) {
      alert('Žádné nalezené kešky k exportu.');
      return;
    }
    // Exportuj pouze pole jmen
    const header = Object.keys(foundCaches[0]);
    const csvRows = [
      header.join(','),
      ...foundCaches.map((row: any) => header.map(field => JSON.stringify(row[field] ?? '')).join(','))
    ];
    const csvContent = csvRows.join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'foundCaches_export.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Import JSON (serverový formát) do waypoints
  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        // Očekáváme JSON pole objektů (serverový formát)
        const data = JSON.parse(text);
        if (!Array.isArray(data)) {
          alert('Soubor neobsahuje platné pole kešek.');
          return;
        }
        localStorage.setItem('waypoints', JSON.stringify(data));
        alert('Import dokončen. Data byla nahrána do aplikace.');
      } catch (err) {
        alert('Chyba při importu: Soubor není platný JSON.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <>
      <IonButton expand="block" color="primary" onClick={handleExport}>
        <IonIcon icon={cloudDownloadOutline} slot="start" />
        Exportovat nalezené kešky (CSV)
      </IonButton>
      <IonButton expand="block" color="secondary" onClick={() => fileInputRef.current?.click()}>
        <IonIcon icon={cloudUploadOutline} slot="start" />
        Importovat kešky ze serverového JSON
      </IonButton>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        style={{ display: 'none' }}
        onChange={handleImport}
      />
    </>
  );
};

export default CsvImportExport;