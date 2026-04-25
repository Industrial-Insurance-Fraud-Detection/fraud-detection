import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import NotificationBell from '../../components/ui/NotificationBell'
import { useDarkMode } from '../../components/layout/Sidebar'

/**
 * InvestigatorDashboard
 *
 * FIX 1 — `claim.client` has firstName + lastName, not fullName.
 * FIX 2 — `claim.equipment` is an object {name, type}.
 * FIX 3 — `claim.claimedAmount` not `claim.amount`.
 * FIX 4 — Score in `claim.analysis?.finalScore`.
 */

const PRIORITY_CONFIG = {
  HIGH: { label: 'Urgent', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  MEDIUM: { label: 'Moyen', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  LOW: { label: 'Faible', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
}

function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

function getPriority(score) {
  if (score >= 60) return 'HIGH'
  if (score >= 40) return 'MEDIUM'
  return 'LOW'
}

// Helper — FIX 1
function clientName(client) {
  return `${client?.firstName || ''} ${client?.lastName || ''}`.trim() || 'Client'
}

function InvestigatorSidebar({ dark }) {
  const navigate = useNavigate()
  const location = useLocation()
  const { logout, user } = useAuthStore()
  const [collapsed, setCollapsed] = useState(false)

  const items = [
    { key: '/investigator/dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: '/investigator/flagged', label: 'Dossiers a traiter', icon: '⚑' },
    { key: '/investigator/history', label: 'Historique', icon: '≡' },
    { key: '/investigator/stats', label: 'Statistiques', icon: '◑' },
    { key: '/investigator/profile', label: 'Mon profil', icon: '👤' },
  ]

  const bg = dark ? '#0A1628' : '#0F2347'
  const border = dark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)'
  const width = collapsed ? 64 : 240
  const active = location.pathname
  // FIX 1 — derive name
  const invName = clientName(user)
  const initial = user?.firstName?.[0]?.toUpperCase() || 'I'

  return (
    <div style={{ width, minHeight: '100vh', backgroundColor: bg, display: 'flex', flexDirection: 'column', position: 'fixed', left: 0, top: 0, zIndex: 100, transition: 'width 0.25s cubic-bezier(0.4,0,0.2,1)', overflow: 'hidden', boxShadow: '4px 0 24px rgba(0,0,0,0.15)' }}>
      <div style={{ padding: collapsed ? '1.5rem 0.75rem' : '1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', fontSize: '1rem', flexShrink: 0 }}>F</div>
            <div>
              <div style={{ color: 'white', fontWeight: 700, fontSize: '0.92rem', whiteSpace: 'nowrap' }}>FraudGuard AI</div>
              <div style={{ color: '#C9A84C', fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: 1 }}>Espace Investigateur</div>
            </div>
          </div>
        )}
        {collapsed && (
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', fontSize: '1rem', margin: '0 auto' }}>F</div>
        )}
        {!collapsed && (
          <button onClick={() => setCollapsed(true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem', padding: '0.25rem', borderRadius: 4 }}
            onMouseEnter={e => e.target.style.color = 'white'} onMouseLeave={e => e.target.style.color = 'rgba(255,255,255,0.3)'}>←</button>
        )}
      </div>

      {collapsed && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '0.5rem 0', borderBottom: `1px solid ${border}` }}>
          <button onClick={() => setCollapsed(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem' }}
            onMouseEnter={e => e.target.style.color = 'white'} onMouseLeave={e => e.target.style.color = 'rgba(255,255,255,0.3)'}>→</button>
        </div>
      )}

      {!collapsed && (
        <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, fontSize: '1rem', flexShrink: 0 }}>
              {initial}
            </div>
            <div>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{invName}</div>
              <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Investigateur senior</div>
            </div>
          </div>
        </div>
      )}
      {collapsed && (
        <div style={{ padding: '0.75rem', display: 'flex', justifyContent: 'center', borderBottom: `1px solid ${border}` }}>
          <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>{initial}</div>
        </div>
      )}

      <nav style={{ flex: 1, padding: '0.75rem 0' }}>
        {items.map(item => {
          const isActive = active === item.key || active.startsWith(item.key + '/')
          return (
            <div key={item.key} onClick={() => navigate(item.key)} title={collapsed ? item.label : ''}
              style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', padding: collapsed ? '0.75rem' : '0.75rem 1.5rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', backgroundColor: isActive ? 'rgba(201,168,76,0.12)' : 'transparent', borderLeft: isActive ? '3px solid #C9A84C' : '3px solid transparent', borderRight: '3px solid transparent', transition: 'all 0.18s' }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'transparent' }}>
              <span style={{ fontSize: '1rem', width: 20, textAlign: 'center', color: isActive ? '#C9A84C' : 'rgba(255,255,255,0.45)', flexShrink: 0 }}>{item.icon}</span>
              {!collapsed && <span style={{ color: isActive ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: isActive ? 600 : 400, flex: 1, whiteSpace: 'nowrap' }}>{item.label}</span>}
            </div>
          )
        })}
      </nav>

      <div style={{ padding: collapsed ? '0.75rem' : '1rem 1.5rem', borderTop: `1px solid ${border}` }}>
        <div onClick={() => { logout(); window.location.href = '/login' }}
          style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', padding: '0.4rem', borderRadius: 6, transition: 'background 0.15s' }}
          onMouseEnter={e => e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'}
          onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
          <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '1rem' }}>↩</span>
          {!collapsed && <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>}
        </div>
      </div>
    </div>
  )
}

export default function InvestigatorDashboard() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const [dark, toggleDark] = useDarkMode()
  const [sortBy, setSortBy] = useState('date')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    api.get('/claims/flagged')
      .then(res => setClaims(extractArray(res.data)))
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      api.get('/claims/flagged')
        .then(res => setClaims(extractArray(res.data)))
        .catch(() => { })
    }, 15000)
    return () => clearInterval(interval)
  }, [])

  const handleSort = (field) => {
    if (sortBy === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortBy(field); setSortDir('desc') }
  }
  const sortIcon = (field) => sortBy !== field ? ' ↕' : sortDir === 'asc' ? ' ↑' : ' ↓'

  const filtered = claims
    .filter(c => {
      // FIX 4 — score from analysis
      const score = c.analysis?.finalScore ?? 50
      const priority = getPriority(score)
      const matchFilter = filter === 'ALL' || priority === filter
      // FIX 1+2 — client name and equipment object
      const cName = clientName(c.client)
      const eqName = c.equipment?.name || ''
      const matchSearch =
        c.reference?.toLowerCase().includes(search.toLowerCase()) ||
        cName.toLowerCase().includes(search.toLowerCase()) ||
        eqName.toLowerCase().includes(search.toLowerCase())
      return matchFilter && matchSearch
    })
    .sort((a, b) => {
      let valA, valB
      if (sortBy === 'date') {
        valA = new Date(a.createdAt).getTime()
        valB = new Date(b.createdAt).getTime()
      } else if (sortBy === 'score') {
        // FIX 4
        valA = a.analysis?.finalScore ?? 0
        valB = b.analysis?.finalScore ?? 0
      } else if (sortBy === 'amount') {
        // FIX 3
        valA = a.claimedAmount ?? 0
        valB = b.claimedAmount ?? 0
      }
      return sortDir === 'asc' ? valA - valB : valB - valA
    })

  const stats = {
    total: claims.length,
    urgent: claims.filter(c => getPriority(c.analysis?.finalScore ?? 50) === 'HIGH').length,
    // FIX 4
    avgScore: claims.length > 0
      ? Math.round(claims.reduce((s, c) => s + (c.analysis?.finalScore ?? 50), 0) / claims.length)
      : 0,
    // FIX 3
    totalAmount: claims.reduce((s, c) => s + (c.claimedAmount ?? 0), 0),
  }

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody = dark ? '#C8D8E8' : '#4B5563'
  const rowHover = dark ? '#172338' : '#F9FAFB'

  const thStyle = (field) => ({
    fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase',
    letterSpacing: '0.06em', color: sortBy === field ? '#C9A84C' : textSub,
    fontFamily: 'Helvetica Neue, Arial, sans-serif',
    cursor: 'pointer', userSelect: 'none', transition: 'color 0.15s',
  })

  // FIX 1 — investigator first name
  const firstName = user?.firstName || 'Investigateur'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem', transition: 'margin 0.25s' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Espace Investigateur</p>
            {/* FIX 1 */}
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>
              Bonjour, <strong>{firstName}</strong> 🔍
            </h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {new Date().toLocaleDateString('fr-FR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark}
              style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
          {[
            { label: 'En attente', value: stats.total, sub: 'Dossiers a traiter', color: '#7D6608' },
            { label: 'Urgents', value: stats.urgent, sub: 'Priorite haute', color: '#C0392B' },
            { label: 'Score moyen', value: stats.avgScore, sub: 'Score fraude moyen', color: '#1A5276' },
            { label: 'Montant total', value: stats.totalAmount > 0 ? `${(stats.totalAmount / 1000000).toFixed(1)}M DA` : '0 DA', sub: 'Valeur en revision', color: '#0F2347' },
          ].map(s => (
            <div key={s.label} style={{ backgroundColor: cardBg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${cardBorder}`, flex: 1, transition: 'transform 0.18s', cursor: 'default' }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
              <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>{s.label}</div>
              <div style={{ fontSize: '2.2rem', fontWeight: 700, color: s.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s.value}</div>
              <div style={{ fontSize: '0.75rem', color: textSub, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{s.sub}</div>
            </div>
          ))}
        </div>

        {/* Table */}
        <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>
          <div style={{ padding: '1.25rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <div>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.1rem' }}>Dossiers en revision humaine</h2>
              <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{filtered.length} dossier(s)</p>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <input placeholder="Rechercher..." value={search} onChange={e => setSearch(e.target.value)}
                style={{ padding: '0.5rem 0.9rem', border: `1.5px solid ${cardBorder}`, borderRadius: 6, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', width: 180, backgroundColor: cardBg, color: textMain }} />
              {[['ALL', 'Tous'], ['HIGH', 'Urgent'], ['MEDIUM', 'Moyen'], ['LOW', 'Faible']].map(([f, l]) => (
                <button key={f} onClick={() => setFilter(f)}
                  style={{ padding: '0.4rem 0.85rem', border: `1.5px solid ${filter === f ? '#0F2347' : cardBorder}`, borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: filter === f ? '#0F2347' : cardBg, color: filter === f ? 'white' : textSub, fontWeight: filter === f ? 600 : 400 }}>
                  {l}
                </button>
              ))}
            </div>
          </div>

          {/* Header row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.5fr 1.8fr 0.9fr 1.1fr 0.9fr 0.9fr 90px', padding: '0.75rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderBottom: `1px solid ${cardBorder}` }}>
            <div style={thStyle(null)}>Reference</div>
            <div style={thStyle(null)}>Client</div>
            <div style={thStyle(null)}>Equipement</div>
            <div style={{ ...thStyle('date'), display: 'flex', alignItems: 'center' }} onClick={() => handleSort('date')}>Date{sortIcon('date')}</div>
            <div style={{ ...thStyle('amount'), display: 'flex', alignItems: 'center' }} onClick={() => handleSort('amount')}>Montant{sortIcon('amount')}</div>
            <div style={{ ...thStyle('score'), display: 'flex', alignItems: 'center' }} onClick={() => handleSort('score')}>Score IA{sortIcon('score')}</div>
            <div style={thStyle(null)}>Priorite</div>
            <div style={thStyle(null)}>Action</div>
          </div>

          {loading && <div style={{ padding: '3rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>}

          {!loading && filtered.length === 0 && (
            <div style={{ padding: '4rem', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>✅</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600 }}>
                {claims.length === 0 ? 'Aucun dossier en revision' : 'Aucun dossier trouve'}
              </div>
              <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                {claims.length === 0 ? 'Tous les sinistres ont ete traites' : 'Modifiez vos filtres'}
              </div>
            </div>
          )}

          {filtered.map((claim, i) => {
            // FIX 4
            const score = claim.analysis?.finalScore ?? 50
            const priority = getPriority(score)
            const pc = PRIORITY_CONFIG[priority]
            // FIX 1
            const cName = clientName(claim.client)
            // FIX 2
            const eqName = claim.equipment?.name || '-'
            return (
              <div key={claim.id}
                style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.5fr 1.8fr 0.9fr 1.1fr 0.9fr 0.9fr 90px', padding: '1rem 1.5rem', borderBottom: i < filtered.length - 1 ? `1px solid ${cardBorder}` : 'none', alignItems: 'center', transition: 'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                <div>
                  {/* FIX 1 */}
                  <div style={{ fontSize: '0.82rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{cName}</div>
                  <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.client?.company || ''}</div>
                </div>
                {/* FIX 2 */}
                <div style={{ fontSize: '0.82rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{eqName}</div>
                <div style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(claim.incidentDate).toLocaleDateString('fr-FR')}</div>
                {/* FIX 3 */}
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.claimedAmount?.toLocaleString('fr-FR')} DA</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <div style={{ flex: 1, height: 5, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${score}%`, backgroundColor: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3 }} />
                  </div>
                  <span style={{ fontSize: '0.78rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', minWidth: 22 }}>{Math.round(score)}</span>
                </div>
                <div>
                  <span style={{ padding: '0.2rem 0.65rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: pc.bg, color: pc.color, border: `1px solid ${pc.border}` }}>{pc.label}</span>
                </div>
                <button onClick={() => navigate(`/investigator/review/${claim.id}`)}
                  style={{ padding: '0.45rem 0.9rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', transition: 'transform 0.15s' }}
                  onMouseEnter={e => e.target.style.transform = 'scale(1.05)'}
                  onMouseLeave={e => e.target.style.transform = 'scale(1)'}>
                  Traiter →
                </button>
              </div>
            )
          })}
        </div>

        <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: cardBg, borderRadius: 10, border: `1px solid ${cardBorder}`, fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          ℹ Ces dossiers necessitent votre intervention. Votre decision est finale et irreversible.
        </div>
      </div>
    </div>
  )
}