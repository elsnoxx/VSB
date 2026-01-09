import React from "react";
import { Link } from "react-router-dom";

export function Navigation() {
  return (
    <nav className="vf-nav">
      <div className="vf-nav-inner">
        <Link to="/devices" className="vf-nav-link">Devices</Link>
        <Link to="/locations" className="vf-nav-link">Locations</Link>
        <Link to="/device-types/create" className="vf-nav-link">Create Device Type</Link>
      </div>
    </nav>
  );
}