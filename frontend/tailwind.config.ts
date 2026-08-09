import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: { glow: "0 12px 38px rgba(99, 102, 241, .22)" },
      colors: { canvas: "#0B1120", panel: "#151B2F", primary: "#6366F1" },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
} satisfies Config;
