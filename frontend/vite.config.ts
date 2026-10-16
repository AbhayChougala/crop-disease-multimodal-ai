import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // strictPort: fail loudly instead of moving to a port the backend's CORS list doesn't allow.
  server: { port: 5173, strictPort: true },
});
