import { Navigate, Route, Routes } from "react-router-dom";

import { Navbar } from "./components/Navbar";
import { AboutPage } from "./pages/AboutPage";
import { AgentsPage } from "./pages/AgentsPage";
import { AskCropAIPage } from "./pages/AskCropAIPage";
import { DiagnosePage } from "./pages/DiagnosePage";
import { HistoryPage } from "./pages/HistoryPage";
import { ProjectPage } from "./pages/ProjectPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { SavedDiagnosisPage } from "./pages/SavedDiagnosisPage";
import { HomePage } from "./pages/HomePage";
import { LiveCameraPage } from "./pages/LiveCameraPage";
import { ResearchPage } from "./pages/ResearchPage";

export default function App() {
  return <><Navbar /><Routes><Route path="/" element={<HomePage />} /><Route path="/diagnose" element={<DiagnosePage />} /><Route path="/live" element={<LiveCameraPage />} /><Route path="/ask" element={<AskCropAIPage />} /><Route path="/agents" element={<AgentsPage />} /><Route path="/history" element={<HistoryPage />} /><Route path="/history/:diagnosisId" element={<SavedDiagnosisPage />} /><Route path="/projects" element={<ProjectsPage />} /><Route path="/projects/:projectId" element={<ProjectPage />} /><Route path="/research" element={<ResearchPage />} /><Route path="/about" element={<AboutPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes><footer><span>Crop Disease Multimodal AI</span><span>Academic research prototype · Not a substitute for expert advice</span></footer></>;
}
