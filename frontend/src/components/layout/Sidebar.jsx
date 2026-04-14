import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'

// Hook mode sombre global
export function useDarkMode() {
  const [dark, setDark] = useState(() => localStorage.getItem('darkMode') === 'true')
  const toggle = () => {
    setDark(d => {
      localStorage.setItem('darkMode', !d)
      return !d
    })
  }
  return [dark, toggle]
}

const CLIENT_ITEMS = [
  { key: '/client/dashboard',  label: 'Tableau de bord',  icon: '▦' },
  { key: '/client/new-claim',  label: 'Nouveau sinistre', icon: '+' },
  { key: '/client/claims',     label: 'Mes sinistres',    icon: '≡' },
  { key: '/client/stats',      label: 'Statistiques',     icon: '◉' },
  { key: '/client/profile',    label: 'Mon profil',       icon: '👤' },
]

const INVESTIGATOR_ITEMS = [
  { key: '/investigator/dashboard', label: 'Tableau de bord',    icon: '▦' },
  { key: '/investigator/flagged',   label: 'Dossiers a traiter', icon: '⚑', badge: true },
  { key: '/investigator/history',   label: 'Historique',         icon: '≡' },
]

export default function Sidebar({ role = 'CLIENT', badgeCount = 0, dark = false }) {
  const navigate = useNavigate()
  const location = useLocation()
  const { logout, user } = useAuthStore()
  const [collapsed, setCollapsed] = useState(false)

  const items = role === 'CLIENT' ? CLIENT_ITEMS : INVESTIGATOR_ITEMS
  const active = location.pathname

  const bg     = dark ? '#0A1628' : '#0F2347'
  const border = dark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)'
  const width  = collapsed ? 64 : 240

  const fullName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Utilisateur'
  const initiale = user?.firstName?.[0]?.toUpperCase() || 'U'

  return (
    <div style={{
      width, minHeight: '100vh', backgroundColor: bg,
      display: 'flex', flexDirection: 'column',
      position: 'fixed', left: 0, top: 0, zIndex: 100,
      transition: 'width 0.25s cubic-bezier(0.4,0,0.2,1)',
      overflow: 'hidden',
      boxShadow: '4px 0 24px rgba(0,0,0,0.15)'
    }}>

      {/* Logo + collapse button */}
      <div style={{ padding: collapsed ? '1.5rem 0.75rem' : '1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', fontSize: '1rem', flexShrink: 0 }}>F</div>
            <div>
              <div style={{ color: 'white', fontWeight: 700, fontSize: '0.92rem', whiteSpace: 'nowrap' }}>FraudGuard AI</div>
              <div style={{ color: '#C9A84C', fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: 1 }}>
                {role === 'CLIENT' ? 'Espace Client' : 'Espace Investigateur'}
              </div>
            </div>
          </div>
        )}
        {collapsed && (
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', fontSize: '1rem', margin: '0 auto' }}>F</div>
        )}
        {!collapsed && (
          <button onClick={() => setCollapsed(true)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem', padding: '0.25rem', borderRadius: 4, transition: 'color 0.2s' }}
            onMouseEnter={e => e.target.style.color = 'white'}
            onMouseLeave={e => e.target.style.color = 'rgba(255,255,255,0.3)'}>
            ←
          </button>
        )}
      </div>

      {/* Expand button quand collapsed */}
      {collapsed && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '0.5rem 0', borderBottom: `1px solid ${border}` }}>
          <button onClick={() => setCollapsed(false)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem', padding: '0.25rem' }}
            onMouseEnter={e => e.target.style.color = 'white'}
            onMouseLeave={e => e.target.style.color = 'rgba(255,255,255,0.3)'}>
            →
          </button>
        </div>
      )}

      {/* User info — expanded */}
      {!collapsed && (
        <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, fontSize: '1rem', flexShrink: 0 }}>
              {initiale}
            </div>
            <div style={{ overflow: 'hidden' }}>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fullName}</div>
              <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', whiteSpace: 'nowrap' }}>
                {role === 'CLIENT' ? (user?.company || 'Client') : 'Investigateur'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* User info — collapsed */}
      {collapsed && (
        <div style={{ padding: '0.75rem', display: 'flex', justifyContent: 'center', borderBottom: `1px solid ${border}` }}>
          <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>
            {initiale}
          </div>
        </div>
      )}

      {/* Nav */}
      <nav style={{ flex: 1, padding: '0.75rem 0', overflowY: 'auto' }}>
        {items.map(item => {
          const isActive = active === item.key || active.startsWith(item.key + '/')
          return (
            <div key={item.key}
              onClick={() => navigate(item.key)}
              title={collapsed ? item.label : ''}
              style={{
                display: 'flex', alignItems: 'center',
                gap: collapsed ? 0 : '0.75rem',
                padding: collapsed ? '0.75rem' : '0.75rem 1.5rem',
                justifyContent: collapsed ? 'center' : 'flex-start',
                cursor: 'pointer',
                backgroundColor: isActive ? 'rgba(201,168,76,0.12)' : 'transparent',
                borderLeft: isActive ? '3px solid #C9A84C' : '3px solid transparent',
                borderRight: '3px solid transparent',
                transition: 'all 0.18s',
                position: 'relative'
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'transparent' }}>
              <span style={{ fontSize: '1rem', width: 20, textAlign: 'center', color: isActive ? '#C9A84C' : 'rgba(255,255,255,0.45)', flexShrink: 0 }}>{item.icon}</span>
              {!collapsed && (
                <>
                  <span style={{ color: isActive ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: isActive ? 600 : 400, flex: 1, whiteSpace: 'nowrap' }}>{item.label}</span>
                  {item.badge && badgeCount > 0 && (
                    <span style={{ backgroundColor: '#C0392B', color: 'white', fontSize: '0.65rem', fontWeight: 700, borderRadius: 10, padding: '0.1rem 0.4rem', flexShrink: 0 }}>{badgeCount}</span>
                  )}
                </>
              )}
              {collapsed && item.badge && badgeCount > 0 && (
                <span style={{ position: 'absolute', top: 8, right: 8, width: 8, height: 8, borderRadius: '50%', backgroundColor: '#C0392B' }} />
              )}
            </div>
          )
        })}
      </nav>

      {/* Footer */}
      <div style={{ padding: collapsed ? '0.75rem' : '1rem 1.5rem', borderTop: `1px solid ${border}` }}>
        <div onClick={() => { logout(); window.location.href = '/login' }}
          style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', padding: '0.4rem', borderRadius: 6, transition: 'background 0.15s' }}
          title={collapsed ? 'Deconnexion' : ''}
          onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'}
          onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
          <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '1rem' }}>↩</span>
          {!collapsed && <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>}
        </div>
      </div>
    </div>
  )
}