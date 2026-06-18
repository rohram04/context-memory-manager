import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy makes the SPA same-origin with the backend (localhost:8000),
// so the signed httpOnly session cookie set by the backend "just works".
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
