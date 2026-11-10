import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#0d0e11", 900: "#14161a", 800: "#1c1f25", 700: "#282c34", 600: "#3a3f4a" },
        accent: { DEFAULT: "#5b8cff", soft: "#8fb0ff" },
        chroma: "#229919",
      },
    },
  },
  plugins: [],
};

export default config;
