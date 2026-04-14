import { useState, useRef, useEffect } from 'react'
import { useNotifications } from '../../hooks/useNotifications'

const TYPE_CONFIG = {
  success: { color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', icon: '✅' },
  error:   { color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE', icon: '❌' },
  warning: { color: '#7D6608', bg: '#FEF9E7', border: '#F7DC6F', icon: '⚑' },
  info:    { color: '#1A5276', bg: '#EBF5FB', border: '#AED6F1', icon: 'ℹ' },
}

export default function NotificationBell({ dark = false }) {
  const { notifications, unreadCount, connected, markAsRead, markAllRead, clearAll } = useNotifications()
  const [open, setOpen] = useState(false)
  const ref = useRef()

  // Fermer si click en dehors
  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Toast auto-affichage pour les nouvelles notifications
  const [toast, setToast] = useState(null)
  useEffect(() => {
    if (notifications.length === 0) return
    const latest = notifications[0]
    if (!latest.read) {
      setToast(latest)
      setTimeout(() => setToast(null), 5000)
    }
  }, [notifications.length])

  const cardBg    = dark ? '#111C30' : 'white'
  const cardBorder= dark ? '#1E2D45' : '#EEF0F6'
  const textMain  = dark ? 'white' : '#0F2347'
  const textSub   = dark ? '#5A7A9A' : '#9CA3AF'
  const rowHover  = dark ? '#172338' : '#F9FAFB'

  return (
    <>
      {/* Toast notification */}
      {toast && (
        <div style={{
          position: 'fixed', top: '1.5rem', right: '1.5rem', zIndex: 9999,
          backgroundColor: TYPE_CONFIG[toast.type]?.bg || '#EBF5FB',
          border: `1px solid ${TYPE_CONFIG[toast.type]?.border || '#AED6F1'}`,
          borderRadius: 12, padding: '1rem 1.25rem',
          maxWidth: 360, boxShadow: '0 8px 30px rgba(0,0,0,0.15)',
          animation: 'slideIn 0.3s ease',
          display: 'flex', alignItems: 'flex-start', gap: '0.75rem'
        }}>
          <span style={{ fontSize: '1.2rem', flexShrink: 0 }}>{TYPE_CONFIG[toast.type]?.icon}</span>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: TYPE_CONFIG[toast.type]?.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{toast.title}</div>
            <div style={{ fontSize: '0.78rem', color: '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{toast.message}</div>
          </div>
          <button onClick={() => setToast(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: textSub, fontSize: '1rem', padding: 0, flexShrink: 0 }}>×</button>
        </div>
      )}

      {/* Cloche */}
      <div ref={ref} style={{ position: 'relative' }}>
        <button onClick={() => setOpen(!open)}
          style={{ position: 'relative', background: 'none', border: `1.5px solid ${cardBorder}`, borderRadius: 8, padding: '0.5rem 0.75rem', cursor: 'pointer', backgroundColor: cardBg, color: textMain, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.3rem', transition: 'all 0.2s' }}>
          🔔
          {unreadCount > 0 && (
            <span style={{ position: 'absolute', top: -6, right: -6, backgroundColor: '#C0392B', color: 'white', fontSize: '0.62rem', fontWeight: 700, borderRadius: '50%', width: 18, height: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </button>

        {/* Panneau notifications */}
        {open && (
          <div style={{
            position: 'absolute', top: '110%', right: 0, width: 380,
            backgroundColor: cardBg, border: `1px solid ${cardBorder}`,
            borderRadius: 14, boxShadow: '0 12px 40px rgba(0,0,0,0.15)',
            zIndex: 1000, overflow: 'hidden'
          }}>
            {/* Header */}
            <div style={{ padding: '1rem 1.25rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  Notifications {unreadCount > 0 && <span style={{ color: '#C0392B' }}>({unreadCount})</span>}
                </div>
                <div style={{ fontSize: '0.7rem', color: connected ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: connected ? '#1A7A4A' : '#C0392B', display: 'inline-block' }} />
                  {connected ? 'Connecte en temps reel' : 'Deconnecte'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {unreadCount > 0 && (
                  <button onClick={markAllRead} style={{ fontSize: '0.72rem', color: '#1A5276', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    Tout lire
                  </button>
                )}
                {notifications.length > 0 && (
                  <button onClick={clearAll} style={{ fontSize: '0.72rem', color: '#C0392B', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    Effacer
                  </button>
                )}
              </div>
            </div>

            {/* Liste */}
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              {notifications.length === 0 ? (
                <div style={{ padding: '2.5rem', textAlign: 'center' }}>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🔔</div>
                  <div style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Aucune notification</div>
                </div>
              ) : (
                notifications.map(notif => {
                  const tc = TYPE_CONFIG[notif.type] || TYPE_CONFIG.info
                  return (
                    <div key={notif.id}
                      onClick={() => markAsRead(notif.id)}
                      style={{ padding: '0.9rem 1.25rem', borderBottom: `1px solid ${cardBorder}`, cursor: 'pointer', backgroundColor: notif.read ? 'transparent' : (dark ? 'rgba(201,168,76,0.05)' : 'rgba(15,35,71,0.02)'), transition: 'background 0.15s', display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}
                      onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                      onMouseLeave={e => e.currentTarget.style.backgroundColor = notif.read ? 'transparent' : (dark ? 'rgba(201,168,76,0.05)' : 'rgba(15,35,71,0.02)')}>
                      <div style={{ width: 36, height: 36, borderRadius: '50%', backgroundColor: tc.bg, border: `1px solid ${tc.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1rem', flexShrink: 0 }}>
                        {tc.icon}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.15rem' }}>{notif.title}</div>
                        <div style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.4 }}>{notif.message}</div>
                        <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                          {new Date(notif.createdAt).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                      {!notif.read && (
                        <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#C9A84C', flexShrink: 0, marginTop: 4 }} />
                      )}
                    </div>
                  )
                })
              )}
            </div>
          </div>
        )}
      </div>
    </>
  )
}