import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import { useDarkMode } from './Sidebar'
import api from '../../api/axios'

const NAV_ITEMS = [
    { key: '/investigator/dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: '/investigator/flagged', label: "Dossiers a traiter", icon: '⚑', badge: true },
    { key: '/investigator/history', label: 'Historique', icon: '≡' },
    { key: '/investigator/stats', label: 'Statistiques', icon: '◑' },
    { key: '/investigator/profile', label: 'Mon profil', icon: '👤' },
]

export function InvestigatorSidebar({ dark = false, badgeCount = 0 }) {
    const navigate = useNavigate()
    const location = useLocation()
    const { logout, refreshToken } = useAuthStore()
    const { user } = useAuthStore()
    const [collapsed, setCollapsed] = useState(false)

    const active = location.pathname
    const width = collapsed ? 64 : 240
    const bg = dark ? '#0A1628' : '#0F2347'
    const border = 'rgba(255,255,255,0.08)'

    const fullName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'
    const initial = (user?.firstName?.[0] ?? 'I').toUpperCase()

    // FIX: call backend before clearing local store
    const handleLogout = async () => {
        try {
            await api.post('/auth/logout', { refreshToken })
        } catch {
            // still clear locally if backend call fails
        }
        logout()
        window.location.href = '/login'
    }

    return (
        <div style={{
            width, minHeight: '100vh', backgroundColor: bg,
            display: 'flex', flexDirection: 'column',
            position: 'fixed', left: 0, top: 0, zIndex: 100,
            transition: 'width 0.25s cubic-bezier(0.4,0,0.2,1)',
            overflow: 'hidden', boxShadow: '4px 0 24px rgba(0,0,0,0.18)',
        }}>
            {/* Logo */}
            <div style={{ padding: collapsed ? '1.5rem 0.75rem' : '1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                {!collapsed && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', fontSize: '1rem' }}>F</div>
                        <div>
                            <div style={{ color: 'white', fontWeight: 700, fontSize: '0.92rem', whiteSpace: 'nowrap' }}>FraudGuard AI</div>
                            <div style={{ color: '#C9A84C', fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Espace Investigateur</div>
                        </div>
                    </div>
                )}
                {collapsed && (
                    <div onClick={() => setCollapsed(false)} style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', margin: '0 auto', cursor: 'pointer' }}>F</div>
                )}
                {!collapsed && (
                    <button onClick={() => setCollapsed(true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem' }}>←</button>
                )}
            </div>

            {/* User info */}
            {!collapsed ? (
                <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, flexShrink: 0 }}>{initial}</div>
                    <div style={{ overflow: 'hidden' }}>
                        <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fullName}</div>
                        <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Investigateur senior</div>
                    </div>
                </div>
            ) : (
                <div style={{ padding: '0.75rem', display: 'flex', justifyContent: 'center', borderBottom: `1px solid ${border}` }}>
                    <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>{initial}</div>
                </div>
            )}

            {/* Nav */}
            <nav style={{ flex: 1, padding: '0.75rem 0', overflowY: 'auto' }}>
                {NAV_ITEMS.map(item => {
                    const isActive = active === item.key || active.startsWith(item.key + '/')
                    return (
                        <div key={item.key} onClick={() => navigate(item.key)} title={collapsed ? item.label : ''}
                            style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', padding: collapsed ? '0.75rem' : '0.75rem 1.5rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', backgroundColor: isActive ? 'rgba(201,168,76,0.12)' : 'transparent', borderLeft: isActive ? '3px solid #C9A84C' : '3px solid transparent', borderRight: '3px solid transparent', transition: 'all 0.18s', position: 'relative' }}
                            onMouseEnter={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)' }}
                            onMouseLeave={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'transparent' }}>
                            <span style={{ fontSize: '1rem', width: 20, textAlign: 'center', color: isActive ? '#C9A84C' : 'rgba(255,255,255,0.45)', flexShrink: 0 }}>{item.icon}</span>
                            {!collapsed && (
                                <>
                                    <span style={{ color: isActive ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: isActive ? 600 : 400, flex: 1, whiteSpace: 'nowrap' }}>{item.label}</span>
                                    {item.badge && badgeCount > 0 && (
                                        <span style={{ backgroundColor: '#C0392B', color: 'white', fontSize: '0.65rem', fontWeight: 700, borderRadius: 10, padding: '0.1rem 0.4rem' }}>{badgeCount}</span>
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

            {/* Logout */}
            <div style={{ padding: collapsed ? '0.75rem' : '1rem 1.5rem', borderTop: `1px solid ${border}` }}>
                <div onClick={handleLogout} title={collapsed ? 'Deconnexion' : ''}
                    style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', padding: '0.4rem', borderRadius: 6 }}
                    onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'}
                    onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                    <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '1rem' }}>↩</span>
                    {!collapsed && <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>}
                </div>
            </div>
        </div>
    )
}

export { useDarkMode }