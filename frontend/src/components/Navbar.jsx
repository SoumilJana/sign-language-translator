import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="flex items-center justify-between bg-gray-200 px-6 py-4 shadow">
      <div className="text-2xl font-bold">
        <Link to="/">SignVision</Link>
      </div>
      <div className="flex space-x-6 font-medium">
        <Link to="/">Home</Link>
        <Link to="/">About Us</Link>
        <Link to="/translator">Translator</Link>
        <Link to="/clone">Clone</Link>
      </div>
    </nav>
  );
}
