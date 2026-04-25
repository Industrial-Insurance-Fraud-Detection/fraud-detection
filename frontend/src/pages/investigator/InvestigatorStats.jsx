import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'
import useAuthStore from '../../store/auth.store'

<<<<<<< HEAD
=======
/**
 * InvestigatorStats
 *
 * FIX — `/claims/stats` does NOT exist in the backend.
 *        Removed that call entirely. Stats are now derived
 *        client-side from the `/claims/flagged` response
 *        (which is what the investigator can access).
 *
 *        Note: this endpoint only returns HUMAN_REVIEW claims,
 *        so "approved" / "rejected" totals are not available
 *        without a dedicated backend endpoint. We show only
 *        what we can accurately compute.
 */

function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

function clientName(client) {
  return `${client?.firstName || ''} ${client?.lastName || ''}`.trim() || 'Client'
}

>>>>>>> a259412 (frontend v2 not completed)
function InvestigatorSidebar({ dark }) {
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()
  const [collapsed, setCollapsed] = useState(false)
  const items = [
<<<<<<< HEAD
    { key: '/investigator/dashboard', label: 'Tableau de bord',    icon: '▦' },
    { key: '/investigator/flagged',   label: 'Dossiers a traiter', icon: '⚑' },
    { key: '/investigator/history',   label: 'Historique',         icon: '≡' },
    { key: '/investigator/stats',     label: 'Statistiques',       icon: '◑' },
    { key: '/investigator/profile',   label: 'Mon profil',         icon: '👤' },
=======
    { key: '/investigator/dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: '/investigator/flagged', label: 'Dossiers a traiter', icon: '⚑' },
    { key: '/investigator/history', label: 'Historique', icon: '≡' },
    { key: '/investigator/stats', label: 'Statistiques', icon: '◑' },
    { key: '/investigator/profile', label: 'Mon profil', icon: '👤' },
>>>>>>> a259412 (frontend v2 not completed)
  ]
  const bg = dark ? '#0A1628' : '#0F2347'
  const border = dark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)'
  const width = collapsed ? 64 : 240
  const active = window.location.pathname
<<<<<<< HEAD
=======
  const initial = user?.firstName?.[0]?.toUpperCase() || 'I'
  const invName = clientName(user)

>>>>>>> a259412 (frontend v2 not completed)
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
<<<<<<< HEAD
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>
              {user?.fullName?.[0]?.toUpperCase() || 'I'}
            </div>
            <div>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{user?.fullName}</div>
=======
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>{initial}</div>
            <div>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{invName}</div>
>>>>>>> a259412 (frontend v2 not completed)
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

function StatCard({ label, value, sub, color, dark }) {
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div style={{ backgroundColor: cardBg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${cardBorder}`, flex: 1, transition: 'transform 0.18s' }}
      onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
      <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.6rem' }}>{label}</div>
      <div style={{ fontSize: '2.2rem', fontWeight: 700, color: color || (dark ? 'white' : '#0F2347'), fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '0.75rem', color: textSub, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{sub}</div>}
    </div>
  )
}

function BarChart({ data, dark }) {
  const max = Math.max(...data.map(d => d.value), 1)
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textMain = dark ? 'white' : '#0F2347'
  return (
    <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
<<<<<<< HEAD
      <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Distribution des scores IA</h3>
=======
      <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
        Distribution des scores IA (dossiers en revision)
      </h3>
>>>>>>> a259412 (frontend v2 not completed)
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.5rem', height: 160 }}>
        {data.map((d, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
            <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{d.value}</div>
            <div style={{ width: '100%', backgroundColor: d.color, borderRadius: '4px 4px 0 0', height: `${(d.value / max) * 120}px`, minHeight: 4, transition: 'height 0.6s ease' }} />
            <div style={{ fontSize: '0.65rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', textAlign: 'center' }}>{d.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function InvestigatorStats() {
<<<<<<< HEAD
  const [stats, setStats] = useState(null)
=======
  // FIX — removed /claims/stats call (endpoint doesn't exist)
>>>>>>> a259412 (frontend v2 not completed)
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
<<<<<<< HEAD
    Promise.all([
      api.get('/claims/stats').catch(() => ({ data: {} })),
      api.get('/claims/flagged').catch(() => ({ data: [] })),
    ]).then(([statsRes, claimsRes]) => {
      setStats(statsRes.data)
      setClaims(claimsRes.data)
    }).finally(() => setLoading(false))
  }, [])

  const pageBg    = dark ? '#0D1626' : '#F7F8FC'
  const cardBg    = dark ? '#111C30' : 'white'
  const cardBorder= dark ? '#1E2D45' : '#EEF0F6'
  const textMain  = dark ? 'white' : '#0F2347'
  const textSub   = dark ? '#5A7A9A' : '#9CA3AF'

  const scoreDistrib = [
    { label: '0-20', value: claims.filter(c => (c.finalScore||50) <= 20).length, color: '#1A7A4A' },
    { label: '21-40', value: claims.filter(c => (c.finalScore||50) > 20 && (c.finalScore||50) <= 40).length, color: '#27AE60' },
    { label: '41-60', value: claims.filter(c => (c.finalScore||50) > 40 && (c.finalScore||50) <= 60).length, color: '#F39C12' },
    { label: '61-80', value: claims.filter(c => (c.finalScore||50) > 60 && (c.finalScore||50) <= 80).length, color: '#E67E22' },
    { label: '81-100', value: claims.filter(c => (c.finalScore||50) > 80).length, color: '#C0392B' },
  ]

  const totalDecided = (stats?.approved || 0) + (stats?.rejected || 0)
  const approvalRate = totalDecided > 0 ? Math.round((stats?.approved / totalDecided) * 100) : 0

=======
    api.get('/claims/flagged')
      .then(res => setClaims(extractArray(res.data)))
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  // Derive all stats client-side from flagged claims
  const total = claims.length
  const urgent = claims.filter(c => (c.analysis?.finalScore ?? 50) >= 60).length
  const avgScore = total > 0
    ? Math.round(claims.reduce((s, c) => s + (c.analysis?.finalScore ?? 50), 0) / total)
    : 0
  const totalAmount = claims.reduce((s, c) => s + (c.claimedAmount ?? 0), 0)

  const scoreDistrib = [
    { label: '0-20', value: claims.filter(c => (c.analysis?.finalScore ?? 50) <= 20).length, color: '#1A7A4A' },
    { label: '21-40', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 20 && (c.analysis?.finalScore ?? 50) <= 40).length, color: '#27AE60' },
    { label: '41-60', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 40 && (c.analysis?.finalScore ?? 50) <= 60).length, color: '#F39C12' },
    { label: '61-80', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 60 && (c.analysis?.finalScore ?? 50) <= 80).length, color: '#E67E22' },
    { label: '81-100', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 80).length, color: '#C0392B' },
  ]

>>>>>>> a259412 (frontend v2 not completed)
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Analyse</p>
<<<<<<< HEAD
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Statistiques <strong>globales</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>Vue d'ensemble du systeme de detection</p>
=======
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Statistiques <strong>dossiers en revision</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              Donnees issues des sinistres en cours d'examen humain
            </p>
>>>>>>> a259412 (frontend v2 not completed)
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
        ) : (
          <>
<<<<<<< HEAD
            {/* Stats cards */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
              <StatCard dark={dark} label="Total sinistres"   value={stats?.total || 0}        sub="Tous statuts"         />
              <StatCard dark={dark} label="Approuves"         value={stats?.approved || 0}     sub="Auto + manuel"        color="#1A7A4A" />
              <StatCard dark={dark} label="Rejetes"           value={stats?.rejected || 0}     sub="Fraude detectee"      color="#C0392B" />
              <StatCard dark={dark} label="En revision"       value={stats?.humanReview || 0}  sub="A traiter"            color="#7D6608" />
              <StatCard dark={dark} label="Taux approbation" value={`${approvalRate}%`}        sub="Sur decisions prises" color="#2E86C1" />
            </div>

            {/* Graphiques */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <BarChart data={scoreDistrib} dark={dark} />

              {/* Repartition statuts */}
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Repartition des statuts</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {[
                    { label: 'Approuves',   value: stats?.approved || 0,    total: stats?.total || 1, color: '#1A7A4A' },
                    { label: 'Rejetes',     value: stats?.rejected || 0,    total: stats?.total || 1, color: '#C0392B' },
                    { label: 'En revision', value: stats?.humanReview || 0, total: stats?.total || 1, color: '#F39C12' },
                    { label: 'En attente',  value: stats?.pending || 0,     total: stats?.total || 1, color: '#1A5276' },
                  ].map(item => (
                    <div key={item.label}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                        <span style={{ fontSize: '0.82rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{item.label}</span>
                        <span style={{ fontSize: '0.82rem', fontWeight: 600, color: item.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{item.value} ({Math.round(item.value/item.total*100)}%)</span>
                      </div>
                      <div style={{ height: 8, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${(item.value/item.total)*100}%`, backgroundColor: item.color, borderRadius: 4, transition: 'width 0.8s ease' }} />
=======
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
              <StatCard dark={dark} label="Dossiers en revision" value={total} sub="Statut HUMAN_REVIEW" />
              <StatCard dark={dark} label="Urgents (score ≥ 60)" value={urgent} sub="Priorite haute" color="#C0392B" />
              <StatCard dark={dark} label="Score IA moyen" value={avgScore} sub="Sur dossiers en cours" color="#1A5276" />
              <StatCard dark={dark} label="Montant total" value={totalAmount > 0 ? `${(totalAmount / 1000000).toFixed(1)}M DA` : '0 DA'} sub="Valeur en revision" color="#2E86C1" />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <BarChart data={scoreDistrib} dark={dark} />

              {/* Score bands legend */}
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Seuils de decision automatique</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {[
                    { range: '0 – 29', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', label: 'Auto-approuve', desc: 'Score faible — aucun indicateur de fraude' },
                    { range: '30 – 69', color: '#F39C12', bg: '#FEF9E7', border: '#F7DC6F', label: 'Revision humaine', desc: 'Zone grise — votre expertise est requise' },
                    { range: '70 – 100', color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE', label: 'Auto-rejete', desc: 'Score eleve — fraude probable' },
                  ].map(item => (
                    <div key={item.range} style={{ display: 'flex', gap: '1rem', padding: '0.75rem', backgroundColor: item.bg, border: `1px solid ${item.border}`, borderRadius: 8 }}>
                      <div style={{ fontWeight: 700, color: item.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.9rem', minWidth: 70 }}>{item.range}</div>
                      <div>
                        <div style={{ fontWeight: 600, color: item.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>{item.label}</div>
                        <div style={{ color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.75rem', marginTop: 2 }}>{item.desc}</div>
>>>>>>> a259412 (frontend v2 not completed)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

<<<<<<< HEAD
            {/* Info systeme IA */}
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
              <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Performance du systeme IA</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Modele 1', name: 'Anomalie LSTM', status: 'Actif', color: '#1A7A4A', poids: '35%' },
                  { label: 'Modele 2', name: 'XGBoost',       status: 'Bientot', color: '#F39C12', poids: '25%' },
                  { label: 'Modele 3', name: 'BERT NLP',      status: 'Bientot', color: '#F39C12', poids: '20%' },
                  { label: 'Modele 4', name: 'YOLOv8 Vision', status: 'Bientot', color: '#F39C12', poids: '20%' },
                ].map(m => (
                  <div key={m.label} style={{ padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}` }}>
                    <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>{m.label} — {m.poids}</div>
                    <div style={{ fontSize: '0.88rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.4rem' }}>{m.name}</div>
                    <span style={{ padding: '0.2rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: m.color === '#1A7A4A' ? '#F0FAF4' : '#FEF9E7', color: m.color, border: `1px solid ${m.color === '#1A7A4A' ? '#B8E4CA' : '#F7DC6F'}` }}>
=======
            {/* AI models info */}
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
              <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Modeles IA du systeme</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Modele 1', name: 'Isolation Forest + LSTM', poids: '35%', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', status: 'Actif' },
                  { label: 'Modele 2', name: 'XGBoost Classification', poids: '25%', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', status: 'Actif' },
                  { label: 'Modele 3', name: 'BERT NLP multilingue', poids: '20%', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', status: 'Actif' },
                  { label: 'Modele 4', name: 'YOLOv8 + ELA Vision', poids: '20%', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', status: 'Actif' },
                ].map(m => (
                  <div key={m.label} style={{ padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}` }}>
                    <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>{m.label} — {m.poids}</div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.4rem' }}>{m.name}</div>
                    <span style={{ padding: '0.2rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: m.bg, color: m.color, border: `1px solid ${m.border}` }}>
>>>>>>> a259412 (frontend v2 not completed)
                      {m.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}