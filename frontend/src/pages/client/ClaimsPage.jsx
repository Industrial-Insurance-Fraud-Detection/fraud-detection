import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'

<<<<<<< HEAD
const STATUS_CONFIG = {
  APPROVED:     { label: 'Approuve',         bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED:     { label: 'Rejete',           bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING:      { label: 'En attente',       bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING:    { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
=======
/**
 * ClaimsPage
 *
 * FIX 1 — `claim.claimedAmount` not `claim.amount`.
 * FIX 2 — `claim.equipment` is an object; use `claim.equipment?.name`.
 * FIX 3 — Score lives in `claim.analysis?.finalScore`.
 */

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuve', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED: { label: 'Rejete', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
>>>>>>> a259412 (frontend v2 not completed)
  HUMAN_REVIEW: { label: 'Revision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
}

export default function ClaimsPage() {
  const navigate = useNavigate()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [filterStatus, setFilterStatus] = useState('ALL')
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    api.get('/claims/my')
      .then(res => {
<<<<<<< HEAD
        const data = res.data?.data?.items || res.data?.data || res.data || []
=======
        const data = res.data?.data?.data || res.data?.data || res.data || []
>>>>>>> a259412 (frontend v2 not completed)
        setClaims(Array.isArray(data) ? data : [])
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const filtered = claims.filter(c => {
<<<<<<< HEAD
    const equipName = c.equipment?.name || c.equipment || ''
    const matchSearch = c.reference?.toLowerCase().includes(search.toLowerCase()) ||
=======
    // FIX 2 — equipment is object
    const equipName = c.equipment?.name || ''
    const matchSearch =
      c.reference?.toLowerCase().includes(search.toLowerCase()) ||
>>>>>>> a259412 (frontend v2 not completed)
      equipName.toLowerCase().includes(search.toLowerCase())
    const matchStatus = filterStatus === 'ALL' || c.status === filterStatus
    return matchSearch && matchStatus
  })

<<<<<<< HEAD
  const pageBg     = dark ? '#0D1626' : '#F7F8FC'
  const cardBg     = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain   = dark ? 'white' : '#0F2347'
  const textSub    = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody   = dark ? '#C8D8E8' : '#4B5563'
  const rowHover   = dark ? '#172338' : '#F9FAFB'
=======
  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody = dark ? '#C8D8E8' : '#4B5563'
  const rowHover = dark ? '#172338' : '#F9FAFB'
>>>>>>> a259412 (frontend v2 not completed)

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Sinistres</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Mes <strong>sinistres</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>{claims.length} sinistre(s) au total</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
            <button onClick={() => navigate('/client/new-claim')}
              style={{ padding: '0.7rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
              + Nouveau sinistre
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input placeholder="Rechercher par reference ou equipement..."
            value={search} onChange={e => setSearch(e.target.value)}
            style={{ padding: '0.6rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', width: 280, backgroundColor: cardBg, color: textMain }} />
<<<<<<< HEAD
          {[['ALL','Tous'],['APPROVED','Approuves'],['REJECTED','Rejetes'],['HUMAN_REVIEW','Revision'],['ANALYZING','En analyse']].map(([f, l]) => (
=======
          {[['ALL', 'Tous'], ['APPROVED', 'Approuves'], ['REJECTED', 'Rejetes'], ['HUMAN_REVIEW', 'Revision'], ['ANALYZING', 'En analyse']].map(([f, l]) => (
>>>>>>> a259412 (frontend v2 not completed)
            <button key={f} onClick={() => setFilterStatus(f)}
              style={{ padding: '0.4rem 0.85rem', border: `1.5px solid ${filterStatus === f ? '#0F2347' : cardBorder}`, borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: filterStatus === f ? '#0F2347' : cardBg, color: filterStatus === f ? 'white' : textSub, fontWeight: filterStatus === f ? 600 : 400 }}>
              {l}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {loading && <div style={{ padding: '3rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>}

          {!loading && filtered.length === 0 && (
            <div style={{ padding: '4rem', textAlign: 'center', backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}` }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📋</div>
              <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600 }}>Aucun sinistre</div>
            </div>
          )}

          {filtered.map(claim => {
<<<<<<< HEAD
            const sc    = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
            const score = claim.finalScore
            const equipName = claim.equipment?.name || claim.equipment || '-'
            const amount = claim.claimedAmount || claim.amount
=======
            const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
            // FIX 3 — nested analysis
            const score = claim.analysis?.finalScore
            // FIX 2 — equipment object
            const equipName = claim.equipment?.name || '-'
            // FIX 1 — claimedAmount
            const amount = claim.claimedAmount
>>>>>>> a259412 (frontend v2 not completed)
            return (
              <div key={claim.id}
                onClick={() => navigate(`/client/claims/${claim.id}`)}
                style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.25rem 1.5rem', cursor: 'pointer', transition: 'all 0.18s', display: 'grid', gridTemplateColumns: '1fr 2fr 1fr 1fr 1fr 1fr', alignItems: 'center', gap: '1rem' }}
                onMouseEnter={e => { e.currentTarget.style.backgroundColor = rowHover; e.currentTarget.style.transform = 'translateX(4px)' }}
                onMouseLeave={e => { e.currentTarget.style.backgroundColor = cardBg; e.currentTarget.style.transform = 'translateX(0)' }}>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                  <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{new Date(claim.incidentDate).toLocaleDateString('fr-FR')}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.88rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{equipName}</div>
                </div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {amount?.toLocaleString('fr-FR')} DA
                </div>
                <div>
                  <span style={{ padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sc.bg, color: sc.color, border: `1px solid ${sc.border}` }}>
                    {sc.label}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {score !== null && score !== undefined ? (
                    <>
                      <div style={{ flex: 1, height: 6, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${score}%`, backgroundColor: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: '0.78rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', minWidth: 24 }}>{Math.round(score)}</span>
                    </>
                  ) : (
                    <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>En cours...</span>
                  )}
                </div>
                <div style={{ textAlign: 'right', color: textSub, fontSize: '0.8rem' }}>→</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}