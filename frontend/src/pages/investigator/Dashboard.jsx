import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import { InvestigatorSidebar } from '../../components/layout/InvestigatorLayout'
import NotificationBell from '../../components/ui/NotificationBell'
import { useNotifications } from '../../hooks/useNotifications'

function extractArray(data) {
  const inner = data?.data ?? data
  const arr = inner?.data ?? inner
  return Array.isArray(arr) ? arr : []
}

function getPriority(score) {
  if (score >= 60) return 'HIGH'
  if (score >= 40) return 'MEDIUM'
  return 'LOW'
}

function clientName(client) {
  return `${client?.firstName || ''} ${client?.lastName || ''}`.trim() || 'Client'
}

const PRIORITY_CONFIG = {
  HIGH: { label: 'Urgent', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  MEDIUM: { label: 'Moyen', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  LOW: { label: 'Faible', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
}

export default function InvestigatorDashboard() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [dark, toggleDark] = useDarkMode()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const { unreadCount } = useNotifications()

  useEffect(() => {
    api.get('/claims/flagged?limit=100')
      .then(res => setClaims(extractArray(res.data)))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  // Poll every 30s to keep stats fresh
  useEffect(() => {
    const id = setInterval(() => {
      api.get('/claims/flagged?limit=100')
        .then(res => setClaims(extractArray(res.data)))
        .catch(() => { })
    }, 30000)
    return () => clearInterval(id)
  }, [])

  const stats = {
    total: claims.length,
    urgent: claims.filter(c => getPriority(c.analysis?.finalScore ?? 50) === 'HIGH').length,
    medium: claims.filter(c => getPriority(c.analysis?.finalScore ?? 50) === 'MEDIUM').length,
    low: claims.filter(c => getPriority(c.analysis?.finalScore ?? 50) === 'LOW').length,
    avgScore: claims.length > 0 ? Math.round(claims.reduce((s, c) => s + (c.analysis?.finalScore ?? 50), 0) / claims.length) : 0,
    totalAmount: claims.reduce((s, c) => s + (c.claimedAmount ?? 0), 0),
  }

  // 5 most urgent claims for the preview table
  const topUrgent = [...claims]
    .sort((a, b) => (b.analysis?.finalScore ?? 0) - (a.analysis?.finalScore ?? 0))
    .slice(0, 5)

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const rowHover = dark ? '#172338' : '#F9FAFB'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <InvestigatorSidebar dark={dark} badgeCount={unreadCount} />

      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Espace Investigateur</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Bonjour, <strong>{user?.firstName || 'Investigateur'}</strong> 🔍</h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {new Date().toLocaleDateString('fr-FR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark}
              style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {/* KPI cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
          {[
            { label: 'Dossiers en attente', value: stats.total, sub: 'Necessite votre action', color: '#7D6608' },
            { label: 'Urgents', value: stats.urgent, sub: 'Score >= 60 — priorite', color: '#C0392B' },
            { label: 'Score moyen', value: `${stats.avgScore}/100`, sub: 'Fraude moyenne queue', color: '#1A5276' },
            { label: 'Montant total', value: stats.totalAmount > 0 ? `${(stats.totalAmount / 1000000).toFixed(1)}M DA` : '0 DA', sub: 'Valeur en revision', color: '#0F2347' },
          ].map(s => (
            <div key={s.label}
              style={{ backgroundColor: cardBg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${cardBorder}`, transition: 'transform 0.18s, box-shadow 0.18s', cursor: 'default' }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = dark ? '0 8px 24px rgba(0,0,0,0.3)' : '0 8px 24px rgba(15,35,71,0.1)' }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}>
              <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>{s.label}</div>
              <div style={{ fontSize: '2.2rem', fontWeight: 700, color: s.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s.value}</div>
              <div style={{ fontSize: '0.75rem', color: textSub, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{s.sub}</div>
            </div>
          ))}
        </div>

        {/* Priority breakdown */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
          {[
            { label: 'Urgent (score >= 60)', count: stats.urgent, color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE', pct: stats.total > 0 ? Math.round(stats.urgent / stats.total * 100) : 0 },
            { label: 'Moyen (40 – 59)', count: stats.medium, color: '#F39C12', bg: '#FEF9E7', border: '#F7DC6F', pct: stats.total > 0 ? Math.round(stats.medium / stats.total * 100) : 0 },
            { label: 'Faible (< 40)', count: stats.low, color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', pct: stats.total > 0 ? Math.round(stats.low / stats.total * 100) : 0 },
          ].map(p => (
            <div key={p.label} style={{ backgroundColor: cardBg, borderRadius: 12, padding: '1.25rem', border: `1px solid ${cardBorder}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: p.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{p.label}</span>
                <span style={{ fontSize: '1.4rem', fontWeight: 700, color: p.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{p.count}</span>
              </div>
              <div style={{ height: 6, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${p.pct}%`, backgroundColor: p.color, borderRadius: 3, transition: 'width 0.6s ease' }} />
              </div>
              <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.3rem' }}>{p.pct}% du total</div>
            </div>
          ))}
        </div>

        {/* Top 5 urgent — preview table */}
        <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden', marginBottom: '1rem' }}>
          <div style={{ padding: '1.25rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>Dossiers les plus urgents</h2>
              <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: '0.15rem 0 0' }}>Top 5 par score de fraude</p>
            </div>
            <button onClick={() => navigate('/investigator/flagged')}
              style={{ padding: '0.5rem 1.2rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
              Voir tous les dossiers →
            </button>
          </div>

          {/* Column headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.5fr 1.8fr 1fr 0.9fr 90px', padding: '0.6rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderBottom: `1px solid ${cardBorder}` }}>
            {['Reference', 'Client', 'Equipement', 'Montant', 'Score IA', 'Action'].map(h => (
              <div key={h} style={{ fontSize: '0.68rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{h}</div>
            ))}
          </div>

          {loading && <div style={{ padding: '2rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>}

          {!loading && topUrgent.length === 0 && (
            <div style={{ padding: '3rem', textAlign: 'center' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>✅</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>File d'attente vide</div>
              <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginTop: '0.3rem' }}>Tous les sinistres ont ete traites automatiquement</div>
            </div>
          )}

          {topUrgent.map((claim, i) => {
            const score = claim.analysis?.finalScore ?? 50
            const priority = getPriority(score)
            const pc = PRIORITY_CONFIG[priority]
            return (
              <div key={claim.id}
                style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.5fr 1.8fr 1fr 0.9fr 90px', padding: '0.9rem 1.5rem', borderBottom: i < topUrgent.length - 1 ? `1px solid ${cardBorder}` : 'none', alignItems: 'center', transition: 'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                <div>
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                  <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(claim.incidentDate).toLocaleDateString('fr-FR')}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.82rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{clientName(claim.client)}</div>
                  <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.client?.company || ''}</div>
                </div>
                <div style={{ fontSize: '0.82rem', color: dark ? '#C8D8E8' : '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.equipment?.name || '-'}</div>
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {claim.claimedAmount != null ? `${claim.claimedAmount.toLocaleString('fr-FR')} DA` : '—'}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <div style={{ flex: 1, height: 5, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${score}%`, backgroundColor: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3 }} />
                  </div>
                  <span style={{ fontSize: '0.78rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', minWidth: 22 }}>{Math.round(score)}</span>
                </div>
                <button onClick={() => navigate(`/investigator/review/${claim.id}`)}
                  style={{ padding: '0.4rem 0.85rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 6, fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                  Traiter →
                </button>
              </div>
            )
          })}
        </div>

        <div style={{ padding: '0.75rem 1rem', backgroundColor: cardBg, borderRadius: 10, border: `1px solid ${cardBorder}`, fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          Score IA : 0–29 = auto-approuve, 30–69 = revision humaine (vous), 70–100 = auto-rejete.
        </div>
      </div>
    </div>
  )
}