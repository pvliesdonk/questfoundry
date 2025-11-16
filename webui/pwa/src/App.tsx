import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import HomePage from './pages/HomePage';
import ProjectsPage from './pages/ProjectsPage';
import HotWorkspacePage from './pages/HotWorkspacePage';
import ColdStoragePage from './pages/ColdStoragePage';
import ExecutionPage from './pages/ExecutionPage';
import SettingsPage from './pages/SettingsPage';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="projects" element={<ProjectsPage />} />
          <Route path="projects/:projectId/hot" element={<HotWorkspacePage />} />
          <Route path="projects/:projectId/cold" element={<ColdStoragePage />} />
          <Route path="projects/:projectId/execute" element={<ExecutionPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
