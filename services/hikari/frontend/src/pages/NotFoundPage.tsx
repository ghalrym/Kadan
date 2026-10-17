import { Link } from 'react-router'

export default function NotFoundPage() {
  return (
    <div className="empty-state stack">
      <h2>Page not found</h2>
      <Link to="/requests">Back to Requests</Link>
    </div>
  )
}
