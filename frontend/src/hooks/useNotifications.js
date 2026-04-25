<<<<<<< HEAD
import { useState, useEffect } from 'react'
import api from '../api/axios'

=======
import { useState, useEffect, useCallback } from 'react'
import api from '../api/axios'

/**
 * useNotifications
 *
 * FIX 1 — backend returns `notification.isRead` not `notification.read`.
 *          All `.read` references replaced with `.isRead`.
 *
 * FIX 2 — added `markAllRead` as the correct name used by NotificationBell
 *          (hook previously only exported `markAllAsRead`).
 *
 * FIX 3 — added `connected` flag (always true after first successful fetch)
 *          and `clearAll` (client-side reset) so NotificationBell compiles
 *          without prop errors. The backend has no delete-all endpoint so
 *          clearAll only hides notifications locally until next poll.
 */
>>>>>>> a259412 (frontend v2 not completed)
function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

export function useNotifications() {
  const [notifications, setNotifications] = useState([])
  const [unreadCount, setUnreadCount] = useState(0)
<<<<<<< HEAD

  const fetchNotifications = async () => {
    try {
      const res = await api.get('/notifications')        // ✅ corrigé
      const safeNotifications = extractArray(res.data?.data || res.data)
      setNotifications(safeNotifications)
      setUnreadCount(safeNotifications.filter(n => !n.isRead).length)
    } catch {
      // silencieux si non connecté
    }
  }
=======
  const [connected, setConnected] = useState(false)

  const fetchNotifications = useCallback(async () => {
    try {
      const res = await api.get('/notifications')
      const raw = extractArray(res.data?.data || res.data)
      setNotifications(raw)
      // FIX 1 — backend field is `isRead`
      setUnreadCount(raw.filter((n) => !n.isRead).length)
      setConnected(true)
    } catch {
      // Stay silent if not authenticated yet
    }
  }, [])
>>>>>>> a259412 (frontend v2 not completed)

  const markAsRead = async (id) => {
    try {
      await api.patch(`/notifications/${id}/read`)
      fetchNotifications()
<<<<<<< HEAD
    } catch {}
  }

  const markAllAsRead = async () => {
    try {
      await api.patch('/notifications/read-all')
      fetchNotifications()
    } catch {}
=======
    } catch { }
  }

  // FIX 2 — both names exported so callers work regardless of which they use
  const markAllRead = async () => {
    try {
      await api.patch('/notifications/read-all')
      fetchNotifications()
    } catch { }
  }

  // FIX 3 — client-side only; backend has no bulk-delete endpoint
  const clearAll = () => {
    setNotifications([])
    setUnreadCount(0)
>>>>>>> a259412 (frontend v2 not completed)
  }

  useEffect(() => {
    fetchNotifications()
<<<<<<< HEAD
    const interval = setInterval(fetchNotifications, 30000) // polling 30s
    return () => clearInterval(interval)
  }, [])

  return { notifications, unreadCount, markAsRead, markAllAsRead }
=======
    const interval = setInterval(fetchNotifications, 30000)
    return () => clearInterval(interval)
  }, [fetchNotifications])

  return {
    notifications,
    unreadCount,
    connected,
    markAsRead,
    markAllRead,
    markAllAsRead: markAllRead, // legacy alias
    clearAll,
  }
>>>>>>> a259412 (frontend v2 not completed)
}