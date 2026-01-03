import React from "react";
import { Link } from "react-router-dom";

export function Navigation() {
  return (
    <nav className="vf-nav">
      <div className="vf-nav-inner">
        <Link to="/devices" className="vf-nav-link">Devices</Link>
        <Link to="/locations" className="vf-nav-link">Locations</Link>
      </div>
    </nav>
  );
}