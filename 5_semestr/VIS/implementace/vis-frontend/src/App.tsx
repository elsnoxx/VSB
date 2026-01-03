import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Header } from "./components/layout/Header";
import { Navigation } from "./components/layout/Navigation";
import { Footer } from "./components/layout/Footer";

import { HomePage } from "./pages/HomePage";
import { DeviceListPage } from "./pages/devices/DeviceListPage";
import { DeviceCreatePage } from "./pages/devices/DeviceCreatePage";
import { DeviceDetailPage } from "./pages/devices/DeviceDetailPage";
import { LocationListPage } from "./pages/locations/LocationListPage";
import { LocationCreatePage } from "./pages/locations/LocationCreatePage";


function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <Header />
        {/* <Navigation /> */}

        <main style={{ padding: "1rem", flex: 1 }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/devices" element={<DeviceListPage />} />
            <Route path="/devices/create" element={<DeviceCreatePage />} />
            <Route path="/devices/:id" element={<DeviceDetailPage />} />
            <Route path="/locations" element={<LocationListPage />} />
            <Route path="/locations/create" element={<LocationCreatePage />} />
          </Routes>
        </main>

        <Footer />
      </div>
    </BrowserRouter>
  );
}



export default App;
