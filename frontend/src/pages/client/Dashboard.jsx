import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuvé', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA', darkBg: '#0D2B1A' },
  REJECTED: { label: 'Rejeté', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE', darkBg: '#2B0D0D' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F', darkBg: '#2B2408' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1', darkBg: '#0D1E2B' },
  HUMAN_REVIEW: { label: 'Révision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1', darkBg: '#0D1E2B' },
}

function extractClaims(responseData) {
  const inner = responseData?.data ?? responseData
  const arr = inner?.data ?? inner
  return Array.isArray(arr) ? arr : []
}

function StatCard({ label, value, sub, color, dark }) {
  const bg = dark ? '#111C30' : 'white'
  const border = dark ? '#1E2D45' : '#EEF0F6'
  const subClr = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div
      style={{ backgroundColor: bg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${border}`, flex: 1, transition: 'transform 0.18s, box-shadow 0.18s', cursor: 'default' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = dark ? '0 8px 24px rgba(0,0,0,0.3)' : '0 8px 24px rgba(15,35,71,0.1)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}
    >
      <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: subClr, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.6rem' }}>{label}</div>
      <div style={{ fontSize: '2.2rem', fontWeight: 700, color: color || (dark ? 'white' : '#0F2347'), fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '0.75rem', color: subClr, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{sub}</div>}
    </div>
  )
}

// Animated greeting component
function AnimatedGreeting({ name, dark }) {
  const [visible, setVisible] = useState(false)
  const [dateVisible, setDateVisible] = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setVisible(true), 80)
    const t2 = setTimeout(() => setDateVisible(true), 320)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  const textMain = dark ? '#FFFFFF' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const gold = '#C9A84C'

  const today = new Date()
  const dateStr = today.toLocaleDateString('fr-FR', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
  // Capitalise first letter
  const dateCap = dateStr.charAt(0).toUpperCase() + dateStr.slice(1)

  return (
    <div>
      <style>{`
        @keyframes greetFadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes dateFadeIn {
          from { opacity: 0; transform: translateX(-8px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes underlineGrow {
          from { width: 0; }
          to   { width: 100%; }
        }
        @keyframes pulseDot {
          0%,100% { opacity: 0.4; transform: scale(1); }
          50%      { opacity: 1;   transform: scale(1.4); }
        }
      `}</style>

      <p style={{
        fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em',
        color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem',
        opacity: visible ? 1 : 0,
        animation: visible ? 'greetFadeUp 0.55s cubic-bezier(0.22,0.61,0.36,1) both' : 'none',
      }}>
        Tableau de bord
      </p>

      <h1 style={{
        fontSize: '1.9rem', fontWeight: 400, letterSpacing: '-0.02em',
        opacity: visible ? 1 : 0,
        animation: visible ? 'greetFadeUp 0.6s cubic-bezier(0.22,0.61,0.36,1) 0.06s both' : 'none',
        margin: 0, lineHeight: 1.2,
      }}>
        <span style={{ color: textSub, fontWeight: 300 }}>Bonjour,</span>{' '}
        <span style={{ position: 'relative', display: 'inline-block' }}>
          <strong style={{
            color: textMain,
            background: `linear-gradient(135deg, ${gold} 0%, #E8C97A 50%, ${gold} 100%)`,
            backgroundSize: '200% 200%',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            animation: 'gradShift 4s ease infinite',
          }}>
            {name}
          </strong>
          {/* animated underline */}
          <span style={{
            position: 'absolute', bottom: -3, left: 0,
            height: 2, borderRadius: 2,
            background: `linear-gradient(90deg, ${gold}, transparent)`,
            animation: visible ? 'underlineGrow 0.8s cubic-bezier(0.22,0.61,0.36,1) 0.4s both' : 'none',
            width: '100%',
          }} />
        </span>
      </h1>

      {/* Date row with live dot */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem',
        opacity: dateVisible ? 1 : 0,
        animation: dateVisible ? 'dateFadeIn 0.5s ease both' : 'none',
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%', backgroundColor: gold,
          display: 'inline-block', flexShrink: 0,
          animation: 'pulseDot 2.2s ease-in-out infinite',
        }} />
        <p style={{
          color: textSub, fontSize: '0.82rem',
          fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0,
        }}>
          {dateCap}
        </p>
      </div>

      <style>{`
        @keyframes gradShift {
          0%   { background-position: 0% 50%; }
          50%  { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </div>
  )
}

export default function ClientDashboard() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  const fetchClaims = () =>
    api.get('/claims/my')
      .then(res => setClaims(extractClaims(res.data)))
      .catch(err => console.error('Dashboard fetch error:', err))
      .finally(() => setLoading(false))

  useEffect(() => { fetchClaims() }, [])

  useEffect(() => {
    const hasActive = claims.some(c => c.status === 'ANALYZING' || c.status === 'PENDING')
    if (!hasActive) return
    const id = setInterval(() => {
      api.get('/claims/my').then(res => setClaims(extractClaims(res.data))).catch(() => { })
    }, 10000)
    return () => clearInterval(id)
  }, [claims])

  const stats = {
    total: claims.length,
    approved: claims.filter(c => c.status === 'APPROVED').length,
    rejected: claims.filter(c => c.status === 'REJECTED').length,
    pending: claims.filter(c => ['PENDING', 'ANALYZING', 'HUMAN_REVIEW'].includes(c.status)).length,
    totalAmount: claims.reduce((s, c) => s + (c.claimedAmount || 0), 0),
  }

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody = dark ? '#C8D8E8' : '#4B5563'
  const rowHover = dark ? '#172338' : '#F9FAFB'
  const gold = '#C9A84C'
  const navy = '#0F2347'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />

      <div style={{ marginLeft: 240, flex: 1, padding: '2rem', transition: 'margin 0.25s' }}>

        {/* ── Header ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <AnimatedGreeting name={user?.firstName || 'Client'} dark={dark} />

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />

            {/* Landing-page style dark mode toggle */}
            <button
              onClick={toggleDark}
              style={{
                background: 'none',
                border: `1px solid ${dark ? 'rgba(201,168,76,0.35)' : 'rgba(15,35,71,0.15)'}`,
                borderRadius: 7,
                padding: '0.4rem 0.85rem',
                cursor: 'pointer',
                color: dark ? 'rgba(201,168,76,0.9)' : textSub,
                fontSize: '0.78rem',
                fontFamily: 'Helvetica Neue, Arial, sans-serif',
                transition: 'border-color 0.2s, color 0.2s',
                backgroundColor: dark ? 'rgba(201,168,76,0.06)' : 'transparent',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = gold; e.currentTarget.style.color = gold }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = dark ? 'rgba(201,168,76,0.35)' : 'rgba(15,35,71,0.15)'; e.currentTarget.style.color = dark ? 'rgba(201,168,76,0.9)' : textSub }}
            >
              {dark ? 'Mode clair' : 'Mode sombre'}
            </button>

            <button onClick={() => navigate('/client/new-claim')}
              style={{ padding: '0.7rem 1.5rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', boxShadow: '0 4px 15px rgba(15,35,71,0.3)', transition: 'transform 0.15s' }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
              + Nouveau sinistre
            </button>
          </div>
        </div>

        {/* ── Stat cards ── */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
          <StatCard dark={dark} label="Total sinistres" value={stats.total} sub="Depuis le début" />
          <StatCard dark={dark} label="Approuvés" value={stats.approved} sub="Sinistres validés" color="#1A7A4A" />
          <StatCard dark={dark} label="Rejetés" value={stats.rejected} sub="Fraude détectée" color="#C0392B" />
          <StatCard dark={dark} label="En cours" value={stats.pending} sub="En attente analyse" color="#E67E22" />
          <StatCard dark={dark} label="Montant total"
            value={stats.totalAmount > 0 ? `${(stats.totalAmount / 1000000).toFixed(1)}M DA` : '0 DA'}
            sub="Valeur déclarée" color="#2E86C1" />
        </div>

        {/* ── Claims table ── */}
        <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>

          <div style={{ padding: '1.25rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.1rem' }}>Mes sinistres récents</h2>
              <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claims.length} sinistre(s) enregistré(s)</p>
            </div>
            {claims.some(c => c.status === 'ANALYZING') && (
              <span style={{ padding: '0.3rem 0.8rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: '#EBF5FB', color: '#1A5276', border: '1px solid #AED6F1' }}>
                ⏳ Analyse en cours...
              </span>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 2fr 1fr 1fr 1.2fr 1fr', padding: '0.75rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderBottom: `1px solid ${cardBorder}` }}>
            {['Référence', 'Équipement', 'Date', 'Montant', 'Statut', 'Score IA'].map(h => (
              <div key={h} style={{ fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{h}</div>
            ))}
          </div>

          {loading && (
            <div style={{ padding: '3rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
          )}

          {!loading && claims.length === 0 && (
            <div style={{ padding: '4rem 2rem', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📋</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Aucun sinistre</div>
              <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '1.5rem' }}>Soumettez votre premier sinistre pour commencer</div>
              <button onClick={() => navigate('/client/new-claim')}
                style={{ padding: '0.65rem 1.5rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
                + Nouveau sinistre
              </button>
            </div>
          )}

          {claims.map((claim, i) => {
            const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
            const score = claim.analysis?.finalScore
            const sBg = dark ? sc.darkBg : sc.bg

            return (
              <div
                key={claim.id}
                onClick={() => navigate(`/client/claims/${claim.id}`)}
                style={{ display: 'grid', gridTemplateColumns: '1.5fr 2fr 1fr 1fr 1.2fr 1fr', padding: '1rem 1.5rem', borderBottom: i < claims.length - 1 ? `1px solid ${cardBorder}` : 'none', cursor: 'pointer', transition: 'background 0.15s', alignItems: 'center' }}
                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                <div style={{ fontSize: '0.85rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {claim.equipment?.name || '—'}
                </div>
                <div style={{ fontSize: '0.8rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
                </div>
                <div style={{ fontSize: '0.85rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>
                  {claim.claimedAmount != null ? claim.claimedAmount.toLocaleString('fr-FR') + ' DA' : '—'}
                </div>
                <div>
                  <span style={{ padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sBg, color: sc.color, border: `1px solid ${sc.border}` }}>
                    {sc.label}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {score != null ? (
                    <>
                      <div style={{ flex: 1, height: 6, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${score}%`, backgroundColor: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3, transition: 'width 0.6s ease' }} />
                      </div>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', minWidth: 28 }}>
                        {Math.round(score)}
                      </span>
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
          {[['0-29', '#1A7A4A', 'Auto approuvé'], ['30-69', '#F39C12', 'Révision humaine'], ['70-100', '#C0392B', 'Auto rejeté']].map(([r, c, l]) => (
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