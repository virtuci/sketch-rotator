import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TurnSketch Studio",
  description: "Rotate a 2D sketch or cel keyframe in 3D and get a clean layer back.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
