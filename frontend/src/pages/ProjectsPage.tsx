import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { createProject, listProjects } from "../api/client";
import type { Project } from "../types/api";

/** Projects group several leaves or plants (a field, a greenhouse, an experiment) with their diagnoses. */
export function ProjectsPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    listProjects().then(setProjects).catch((problem) => {
      setProjects([]);
      setError(problem instanceof Error ? problem.message : "Could not load projects. Is the backend running?");
    });
  }, []);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setBusy(true); setError(null);
    try {
      const project = await createProject(name.trim(), description.trim());
      navigate(`/projects/${project.project_id}`);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Could not create the project.");
    } finally { setBusy(false); }
  }

  return (
    <main className="page-shell">
      <section className="page-intro">
        <p className="eyebrow">Local database</p>
        <h1>Projects</h1>
        <p>Group the leaves or plants of one field, greenhouse or experiment. Upload several photos at once; each is stored with its diagnosis and a label, and the project shows how many plants have each disease.</p>
      </section>

      <form className="panel project-create" onSubmit={(event) => void create(event)}>
        <h2>New project</h2>
        <input value={name} maxLength={120} required placeholder="Name, e.g. North field tomatoes" onChange={(event) => setName(event.target.value)} />
        <input value={description} maxLength={1000} placeholder="Description (optional)" onChange={(event) => setDescription(event.target.value)} />
        <button className="button primary" disabled={!name.trim() || busy}>{busy ? "Creating…" : "Create project"}</button>
      </form>
      {error && <p className="chat-error" role="alert">{error}</p>}

      {projects && projects.length === 0 && !error && <p className="muted">No projects yet.</p>}
      <section className="project-list">
        {projects?.map((project) => (
          <Link key={project.project_id} to={`/projects/${project.project_id}`} className="panel project-tile">
            <strong>{project.name}</strong>
            {project.description && <p>{project.description}</p>}
            <small>{project.image_count} photo{project.image_count === 1 ? "" : "s"} · updated {new Date(project.updated_at).toLocaleDateString()}</small>
          </Link>
        ))}
      </section>
    </main>
  );
}
