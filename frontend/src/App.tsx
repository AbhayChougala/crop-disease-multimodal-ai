import { Navigate, Route, Routes } from "react-router-dom";

import { Navbar } from "./components/Navbar";
import { AboutPage } from "./pages/AboutPage";
import { DiagnosePage } from "./pages/DiagnosePage";
import { HomePage } from "./pages/HomePage";
import { LiveCameraPage } from "./pages/LiveCameraPage";
import { ResearchPage } from "./pages/ResearchPage";

export default function App() {
  return <><Navbar /><Routes><Route path="/" element={<HomePage />} /><Route path="/diagnose" element={<DiagnosePage />} /><Route path="/live" element={<LiveCameraPage />} /><Route path="/research" element={<ResearchPage />} /><Route path="/about" element={<AboutPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes><footer><span>Crop Disease Multimodal AI</span><span>Academic research prototype · Not a substitute for expert advice</span></footer></>;
}
