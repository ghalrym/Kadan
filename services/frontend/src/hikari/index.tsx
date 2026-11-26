import './index.css'
import { client } from './api/generated/client.gen'
import App from './App'

// Hikari's API is served under /hikari; generated paths start at /v1.
client.setConfig({ baseUrl: '/hikari' })

export default App
