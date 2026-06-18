/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ide: {
          bg: "#0d1117",
          panel: "#0f1620",
          card: "#161b22",
          cardHover: "#1c2230",
          border: "#222a35",
          borderBright: "#2f3a48",
          text: "#c9d1d9",
          textDim: "#8b949e",
          textFaint: "#5c6672",
        },
        accent: {
          DEFAULT: "#58a6ff",
          glow: "#388bfd",
        },
        diff: {
          added: "#3fb950",
          removed: "#f85149",
          compressed: "#d29922",
          augmented: "#58a6ff",
        },
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.4)",
        elevated: "0 8px 24px rgba(0,0,0,0.5)",
      },
      keyframes: {
        pulseCritical: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
      },
      animation: {
        pulseCritical: "pulseCritical 1.1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
