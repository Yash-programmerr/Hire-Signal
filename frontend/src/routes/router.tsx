import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppShell } from "../components/app-shell";
import { UploadPage } from "../pages/upload-page";
import { ProcessingPage } from "../pages/processing-page";
import { ResultsPage } from "../pages/results-page";

export const router = createBrowserRouter([{ element: <AppShell />, children: [{ index: true, element: <Navigate to="/upload" replace /> }, { path: "/upload", element: <UploadPage /> }, { path: "/processing", element: <ProcessingPage /> }, { path: "/results", element: <ResultsPage /> }, { path: "*", element: <Navigate to="/upload" replace /> }] }]);
