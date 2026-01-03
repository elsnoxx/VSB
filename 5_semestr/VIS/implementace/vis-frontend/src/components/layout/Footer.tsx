import React from "react";

export function Footer() {
  return (
    <footer className="vf-footer">
      <div className="vf-footer-inner">
        <span>© {new Date().getFullYear()} VIS — frontend</span>
        <small className="vf-footer-right">Demo version</small>
      </div>
    </footer>
  );
}