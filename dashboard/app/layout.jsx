import "./globals.css";

export const metadata = {
  title: "CCO Dashboard",
  description: "Cuda-Compute-OSS PR queue and verified evaluation results.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
