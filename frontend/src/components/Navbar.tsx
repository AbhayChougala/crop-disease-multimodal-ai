import { NavLink } from "react-router-dom";

const links = [
  ["/", "Home"],
  ["/diagnose", "Diagnose"],
  ["/live", "Live Camera"],
  ["/ask", "Ask CropAI"],
  ["/research", "Research"],
  ["/about", "About"],
];

export function Navbar() {
  return (
    <header className="navbar">
      <NavLink className="brand" to="/" aria-label="Home">
        <span className="brand-mark">C</span>
        <span>CropAI Research</span>
      </NavLink>
      <nav aria-label="Primary navigation">
        {links.map(([to, label]) => (
          <NavLink key={to} to={to} className={({ isActive }) => (isActive ? "active" : "")}>
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}
