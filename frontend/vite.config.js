import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api/* to the FastAPI backend during local development. The
// backend itself serves its routes under /api/* (see backend/app/main.py),
// so no path rewriting is needed here -- this also keeps dev, docker-compose,
// and the single-container deployment all using the exact same paths.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
