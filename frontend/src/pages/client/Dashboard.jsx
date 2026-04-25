import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

/**
 * ClientDashboard
 *
 * FIX 1 — `claim.claimedAmount` (not `claim.amount`) — backend field name.
 * FIX 2 — `claim.equipment` is an object {name, type}; use `claim.equipment?.name`.
 * FIX 3 — AI score lives in `claim.analysis?.finalScore`, not `claim.finalScore`.
 */

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuve', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA', darkBg: '#0D2B1A' },
  REJECTED: { label: 'Rejete', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE', darkBg: '#2B0D0D' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F', darkBg: '#2B2408' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1', darkBg: '#0D1E2B' },
  HUMAN_REVIEW: { label: 'Revision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1', darkBg: '#0D1E2B' },
}

function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

function StatCard({ label, value, sub, color, dark }) {
  const bg = dark ? '#111C30' : 'white'
  const border = dark ? '#1E2D45' : '#EEF0F6'
  const subClr = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div style={{ backgroundColor: bg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${border}`, flex: 1, transition: 'transform 0.18s, box-shadow 0.18s', cursor: 'default' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = dark ? '0 8px 24px rgba(0,0,0,0.3)' : '0 8px 24px rgba(15,35,71,0.1)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}>
      <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: subClr, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.6rem' }}>{label}</div>
      <div style={{ fontSize: '2.2rem', fontWeight: 700, color: color || (dark ? 'white' : '#0F2347'), fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '0.75rem', color: subClr, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{sub}</div>}
    </div>
  )
}

export default function ClientDashboard() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    api.get('/claims/my')
      .then(res => setClaims(extractArray(res.data)))
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!Array.isArray(claims)) return
    const hasAnalyzing = claims.some(c => c.status === 'ANALYZING' || c.status === 'PENDING')
    if (!hasAnalyzing) return
    const interval = setInterval(() => {
      api.get('/claims/my')
        .then(res => setClaims(extractArray(res.data)))
        .catch(() => { })
    }, 10000)
    return () => clearInterval(interval)
  }, [claims])

  const safeClaims = Array.isArray(claims) ? claims : []

  const stats = {
    total: safeClaims.length,
    approved: safeClaims.filter(c => c.status === 'APPROVED').length,
    rejected: safeClaims.filter(c => c.status === 'REJECTED').length,
    pending: safeClaims.filter(c => ['PENDING', 'ANALYZING', 'HUMAN_REVIEW'].includes(c.status)).length,
    // FIX 1 — claimedAmount
    totalAmount: safeClaims.reduce((s, c) => s + (c.claimedAmount || 0), 0),
  }

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody = dark ? '#C8D8E8' : '#4B5563'
  const rowHover = dark ? '#172338' : '#F9FAFB'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem', transition: 'margin 0.25s' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Tableau de bord</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400, letterSpacing: '-0.02em' }}>
              Bonjour, <strong>{user?.firstName || 'Client'}</strong> 👋
            </h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {new Date().toLocaleDateString('fr-FR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark}
              style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub, display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'all 0.2s' }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
            <button onClick={() => navigate('/client/new-claim')}
              style={{ padding: '0.7rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', boxShadow: '0 4px 15px rgba(15,35,71,0.3)', transition: 'transform 0.15s, box-shadow 0.15s' }}
              onMouseEnter={e => { e.target.style.transform = 'translateY(-1px)'; e.target.style.boxShadow = '0 6px 20px rgba(15,35,71,0.4)' }}
              onMouseLeave={e => { e.target.style.transform = 'translateY(0)'; e.target.style.boxShadow = '0 4px 15px rgba(15,35,71,0.3)' }}>
              + Nouveau sinistre
            </button>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
          <StatCard dark={dark} label="Total sinistres" value={stats.total} sub="Depuis le debut" />
          <StatCard dark={dark} label="Approuves" value={stats.approved} sub="Sinistres valides" color="#1A7A4A" />
          <StatCard dark={dark} label="Rejetes" value={stats.rejected} sub="Fraude detectee" color="#C0392B" />
          <StatCard dark={dark} label="En cours" value={stats.pending} sub="En attente analyse" color="#E67E22" />
          <StatCard dark={dark} label="Montant total"
            value={stats.totalAmount > 0 ? `${(stats.totalAmount / 1000000).toFixed(1)}M DA` : '0 DA'}
            sub="Valeur declaree" color="#2E86C1" />
        </div>

        {/* Table */}
        <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden', transition: 'background 0.3s' }}>
          <div style={{ padding: '1.25rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.1rem' }}>Mes sinistres recents</h2>
              <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{safeClaims.length} sinistre(s) enregistre(s)</p>
            </div>
            {safeClaims.some(c => c.status === 'ANALYZING') && (
              <span style={{ padding: '0.3rem 0.8rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: '#EBF5FB', color: '#1A5276', border: '1px solid #AED6F1' }}>
                ⏳ Analyse en cours...
              </span>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 2fr 1fr 1fr 1.2fr 1fr', padding: '0.75rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderBottom: `1px solid ${cardBorder}` }}>
            {['Reference', 'Equipement', 'Date', 'Montant', 'Statut', 'Score IA'].map(h => (
              <div key={h} style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{h}</div>
            ))}
          </div>

          {loading && (
            <div style={{ padding: '3rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
          )}

          {!loading && safeClaims.length === 0 && (
            <div style={{ padding: '4rem 2rem', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📋</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Aucun sinistre</div>
              <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '1.5rem' }}>Soumettez votre premier sinistre pour commencer</div>
              <button onClick={() => navigate('/client/new-claim')}
                style={{ padding: '0.65rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
                + Nouveau sinistre
              </button>
            </div>
          )}

          {safeClaims.map((claim, i) => {
            const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
            // FIX 3 — score nested inside analysis
            const score = claim.analysis?.finalScore
            const sBg = dark ? sc.darkBg : sc.bg
            return (
              <div key={claim.id}
                onClick={() => navigate(`/client/claims/${claim.id}`)}
                style={{ display: 'grid', gridTemplateColumns: '1.5fr 2fr 1fr 1fr 1.2fr 1fr', padding: '1rem 1.5rem', borderBottom: i < safeClaims.length - 1 ? `1px solid ${cardBorder}` : 'none', cursor: 'pointer', transition: 'background 0.15s', alignItems: 'center' }}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                {/* FIX 2 — equipment is an object */}
                <div style={{ fontSize: '0.85rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {claim.equipment?.name || '-'}
                </div>
                <div style={{ fontSize: '0.8rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
                </div>
                {/* FIX 1 — claimedAmount */}
                <div style={{ fontSize: '0.85rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>
                  {claim.claimedAmount?.toLocaleString('fr-FR')} DA
                </div>
                <div>
                  <span style={{ padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sBg, color: sc.color, border: `1px solid ${sc.border}` }}>
                    {sc.label}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {score !== null && score !== undefined ? (
                    <>
                      <div style={{ flex: 1, height: 6, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${score}%`, backgroundColor: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3, transition: 'width 0.6s ease' }} />
                      </div>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', minWidth: 28 }}>{Math.round(score)}</span>
                    </>
                  ) : (
                    <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>En cours...</span>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: cardBg, borderRadius: 10, border: `1px solid ${cardBorder}` }}>
          <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score IA :</span>
          {[['0-29', '#1A7A4A', 'Auto approuve'], ['30-69', '#F39C12', 'Revision humaine'], ['70-100', '#C0392B', 'Auto rejete']].map(([r, c, l]) => (
            <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: c }} />
              <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r} — {l}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}