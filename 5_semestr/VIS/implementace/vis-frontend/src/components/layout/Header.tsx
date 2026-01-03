import React from "react";
import { Link } from "react-router-dom";
import { Navigation } from "./Navigation";

export function Header() {
  return (
    <header className="vf-header">
      <div className="vf-header-inner">
        
        <div className="vf-logo">
          <span className="vf-logo-mark">VIS</span>
          <Link to="/" className="vf-header-logo">
            <div className="vf-logo-text">
              
              <strong>VIS Frontend</strong>
              
            </div>
          </Link>
        </div>
        

        <nav className="vf-nav" aria-label="Main navigation">
          <div className="vf-nav-inner">
            <Navigation />
          </div>
        </nav>

        <div className="vf-header-actions">
          <button className="vf-btn vf-btn-ghost">Sign in</button>
        </div>
      </div>
    </header>
  );
}