import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'
import useAuthStore from '../../store/auth.store'

const STATUS_CONFIG = {
  APPROVED:     { label: 'Approuve',       bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED:     { label: 'Rejete',         bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  HUMAN_REVIEW: { label: 'En revision',    bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING:    { label: 'En analyse',     bg: '#EFF6FF', color: '#1D4ED8', border: '#BFDBFE' },
}

function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

function InvestigatorSidebar({ dark }) {
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()
  const [collapsed, setCollapsed] = useState(false)
  const items = [
    { key: '/investigator/dashboard', label: 'Tableau de bord',    icon: '▦' },
    { key: '/investigator/flagged',   label: 'Dossiers a traiter', icon: '⚑' },
    { key: '/investigator/history',   label: 'Historique',         icon: '≡' },
    { key: '/investigator/stats',     label: 'Statistiques',       icon: '◑' },
    { key: '/investigator/profile',   label: 'Mon profil',         icon: '👤' },
  ]
  const bg     = dark ? '#0A1628' : '#0F2347'
  const border = dark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)'
  const width  = collapsed ? 64 : 240
  const active = window.location.pathname

  return (
    <div style={{ width, minHeight: '100vh', backgroundColor: bg, display: 'flex', flexDirection: 'column', position: 'fixed', left: 0, top: 0, zIndex: 100, transition: 'width 0.25s', overflow: 'hidden', boxShadow: '4px 0 24px rgba(0,0,0,0.15)' }}>
      <div style={{ padding: collapsed ? '1.5rem 0.75rem' : '1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {!collapsed ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347' }}>F</div>
              <div>
                <div style={{ color: 'white', fontWeight: 700, fontSize: '0.92rem' }}>FraudGuard AI</div>
                <div style={{ color: '#C9A84C', fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Espace Investigateur</div>
              </div>
            </div>
            <button onClick={() => setCollapsed(true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem' }}>←</button>
          </>
        ) : (
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', margin: '0 auto', cursor: 'pointer' }} onClick={() => setCollapsed(false)}>F</div>
        )}
      </div>
      {!collapsed && (
        <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>
              {user?.fullName?.[0]?.toUpperCase() || 'I'}
            </div>
            <div>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{user?.fullName}</div>
              <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Investigateur senior</div>
            </div>
          </div>
        </div>
      )}
      <nav style={{ flex: 1, padding: '0.75rem 0' }}>
        {items.map(item => {
          const isActive = active === item.key
          return (
            <div key={item.key} onClick={() => navigate(item.key)}
              style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', padding: collapsed ? '0.75rem' : '0.75rem 1.5rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', backgroundColor: isActive ? 'rgba(201,168,76,0.12)' : 'transparent', borderLeft: isActive ? '3px solid #C9A84C' : '3px solid transparent', transition: 'all 0.18s' }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'transparent' }}>
              <span style={{ color: isActive ? '#C9A84C' : 'rgba(255,255,255,0.45)', width: 20, textAlign: 'center' }}>{item.icon}</span>
              {!collapsed && <span style={{ color: isActive ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: isActive ? 600 : 400 }}>{item.label}</span>}
            </div>
          )
        })}
      </nav>
      <div style={{ padding: collapsed ? '0.75rem' : '1rem 1.5rem', borderTop: `1px solid ${border}` }}>
        <div onClick={() => { logout(); window.location.href = '/login' }} style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', padding: '0.4rem', borderRadius: 6 }}>
          <span style={{ color: 'rgba(255,255,255,0.35)' }}>↩</span>
          {!collapsed && <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>}
        </div>
      </div>
    </div>
  )
}

export default function InvestigatorHistory() {
  const [claims, setClaims]             = useState([])
  const [loading, setLoading]           = useState(true)
  const [search, setSearch]             = useState('')
  const [filterStatus, setFilterStatus] = useState('ALL')
  const [dark, toggleDark]              = useDarkMode()
  const navigate                        = useNavigate()

  useEffect(() => {
    api.get('/claims/flagged')
      .then(res => setClaims(extractArray(res.data)))
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const filtered = claims.filter(claim => {
    const matchStatus = filterStatus === 'ALL' || claim.status === filterStatus
    const q = search.toLowerCase()
    const matchSearch = !q
      || claim.reference?.toLowerCase().includes(q)
      || claim.equipment?.toLowerCase().includes(q)
      || claim.client?.fullName?.toLowerCase().includes(q)
      || claim.location?.toLowerCase().includes(q)
    return matchStatus && matchSearch
  })

  const pageBg     = dark ? '#0D1626' : '#F7F8FC'
  const cardBg     = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain   = dark ? 'white' : '#0F2347'
  const textSub    = dark ? '#5A7A9A' : '#9CA3AF'
  const rowHover   = dark ? '#172338' : '#F9FAFB'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Investigateur</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Historique des <strong>decisions</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {filtered.length} dossier{filtered.length > 1 ? 's' : ''} trouve{filtered.length > 1 ? 's' : ''}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <input
            placeholder="Rechercher par reference, equipement, client..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ padding: '0.6rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', width: 320, backgroundColor: cardBg, color: textMain }}
          />
          {[['ALL','Tous'],['APPROVED','Approuves'],['REJECTED','Rejetes'],['HUMAN_REVIEW','En revision']].map(([f, l]) => (
            <button key={f} onClick={() => setFilterStatus(f)}
              style={{ padding: '0.4rem 0.85rem', border: `1.5px solid ${filterStatus === f ? '#C9A84C' : cardBorder}`, borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: filterStatus === f ? '#C9A84C' : cardBg, color: filterStatus === f ? '#0F2347' : textSub, fontWeight: filterStatus === f ? 700 : 400 }}>
              {l}
            </button>
          ))}
        </div>

        <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>
          <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Dossiers traites</h2>
            <span style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{filtered.length} resultat{filtered.length > 1 ? 's' : ''}</span>
          </div>

          {loading && (
            <div style={{ padding: '3rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
          )}

          {!loading && filtered.length === 0 && (
            <div style={{ padding: '4rem', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🔍</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>Aucun resultat</div>
              <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                {search ? `Aucun dossier pour "${search}"` : 'Aucun dossier dans cette categorie'}
              </div>
            </div>
          )}

          {!loading && filtered.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 2fr 1.5fr 1.2fr 1fr 1fr', padding: '0.6rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderBottom: `1px solid ${cardBorder}` }}>
              {['Reference','Equipement','Client','Montant','Statut','Date'].map(h => (
                <div key={h} style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>{h}</div>
              ))}
            </div>
          )}

          {filtered.map((claim, i) => {
            const sc = STATUS_CONFIG[claim.status] || { label: claim.status, bg: '#F7F8FC', color: '#6B7280', border: '#E5E7EB' }
            return (
              <div key={claim.id}
                onClick={() => navigate(`/investigator/review/${claim.id}`)}
                style={{ display: 'grid', gridTemplateColumns: '1.2fr 2fr 1.5fr 1.2fr 1fr 1fr', padding: '1rem 1.5rem', borderBottom: i < filtered.length - 1 ? `1px solid ${cardBorder}` : 'none', cursor: 'pointer', alignItems: 'center', transition: 'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                <div style={{ fontSize: '0.85rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.equipment}</div>
                <div style={{ fontSize: '0.82rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.client?.fullName || '-'}</div>
                <div style={{ fontSize: '0.82rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>{claim.amount?.toLocaleString('fr-FR')} DA</div>
                <div>
                  <span style={{ padding: '0.25rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sc.bg, color: sc.color, border: `1px solid ${sc.border}`, whiteSpace: 'nowrap' }}>
                    {sc.label}
                  </span>
                </div>
                <div style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {claim.decidedAt
                    ? new Date(claim.decidedAt).toLocaleDateString('fr-FR')
                    : new Date(claim.updatedAt).toLocaleDateString('fr-FR')}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}