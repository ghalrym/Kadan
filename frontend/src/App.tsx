import { BrowserRouter, Navigate, Route, Routes } from 'react-router'
import AppLayout from './components/AppLayout'
import RequestDetails from './components/RequestDetails'
import ApiAccessPage from './pages/ApiAccessPage'
import ChatPage from './pages/ChatPage'
import DecisionsPage from './pages/DecisionsPage'
import ImagePage from './pages/ImagePage'
import NotFoundPage from './pages/NotFoundPage'
import RequestsPage from './pages/RequestsPage'
import SettingsPage from './pages/SettingsPage'
import SpeechToTextPage from './pages/SpeechToTextPage'
import TextToSpeechPage from './pages/TextToSpeechPage'
import VideoPage from './pages/VideoPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<Navigate to="/requests" replace />} />
          <Route path="requests" element={<RequestsPage />}>
            <Route path=":requestId" element={<RequestDetails />} />
          </Route>
          <Route path="chat" element={<ChatPage />} />
          <Route path="decisions" element={<DecisionsPage />} />
          <Route path="image" element={<ImagePage />} />
          <Route path="image/edit" element={<ImagePage edit />} />
          <Route path="video" element={<VideoPage />} />
          <Route path="tts" element={<TextToSpeechPage />} />
          <Route path="stt" element={<SpeechToTextPage />} />
          <Route path="api" element={<ApiAccessPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
