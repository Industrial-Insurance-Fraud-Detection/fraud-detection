import { useState, useEffect } from 'react'
import api from '../api/axios'

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

  const markAsRead = async (id) => {
    try {
      await api.patch(`/notifications/${id}/read`)
      fetchNotifications()
    } catch {}
  }

  const markAllAsRead = async () => {
    try {
      await api.patch('/notifications/read-all')
      fetchNotifications()
    } catch {}
  }

  useEffect(() => {
    fetchNotifications()
    const interval = setInterval(fetchNotifications, 30000) // polling 30s
    return () => clearInterval(interval)
  }, [])

  return { notifications, unreadCount, markAsRead, markAllAsRead }
}