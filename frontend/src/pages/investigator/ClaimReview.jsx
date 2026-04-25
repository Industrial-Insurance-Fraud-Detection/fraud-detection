import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'

<<<<<<< HEAD
=======
/**
 * ClaimReview (Investigator)
 *
 * FIX 1 — Payload to PATCH /claims/:id/decide must be { outcome, notes }
 *          not { decision, note }.
 *
 * FIX 2 — `claim.client` has `firstName` + `lastName`, not `fullName`.
 *          All client name references updated.
 *
 * FIX 3 — `claim.equipment` is an object {name, type}, not a string.
 *
 * FIX 4 — AI scores live in `claim.analysis.*`, not directly on claim.
 *
 * FIX 5 — `claim.claimedAmount` not `claim.amount`.
 */

>>>>>>> a259412 (frontend v2 not completed)
function Sidebar({ active }) {
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()
  const items = [
<<<<<<< HEAD
    { key: 'dashboard', label: 'Tableau de bord',    icon: '▦' },
    { key: 'flagged',   label: 'Dossiers a traiter', icon: '⚑' },
    { key: 'history',   label: 'Historique',         icon: '≡' },
    { key: 'stats',     label: 'Statistiques',       icon: '◑' },
  ]
=======
    { key: 'dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: 'flagged', label: 'Dossiers a traiter', icon: '⚑' },
    { key: 'history', label: 'Historique', icon: '≡' },
    { key: 'stats', label: 'Statistiques', icon: '◑' },
  ]
  // FIX 2 — derive name from firstName/lastName
  const investigatorName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'
>>>>>>> a259412 (frontend v2 not completed)
  return (
    <div style={{ width: 240, minHeight: '100vh', backgroundColor: '#0F2347', display: 'flex', flexDirection: 'column', position: 'fixed', left: 0, top: 0, zIndex: 100 }}>
      <div style={{ padding: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347' }}>F</div>
          <div>
            <div style={{ color: 'white', fontWeight: 700, fontSize: '0.95rem' }}>FraudGuard AI</div>
            <div style={{ color: '#C9A84C', fontSize: '0.62rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Espace Investigateur</div>
          </div>
        </div>
      </div>
      <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
<<<<<<< HEAD
        <div style={{ width: 38, height: 38, borderRadius: '50%', backgroundColor: '#C9A84C', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, marginBottom: '0.5rem' }}>{user?.fullName?.[0] || 'I'}</div>
        <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{user?.fullName || 'Investigateur'}</div>
=======
        <div style={{ width: 38, height: 38, borderRadius: '50%', backgroundColor: '#C9A84C', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, marginBottom: '0.5rem' }}>{user?.firstName?.[0] || 'I'}</div>
        <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{investigatorName}</div>
>>>>>>> a259412 (frontend v2 not completed)
        <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Investigateur senior</div>
      </div>
      <nav style={{ flex: 1, padding: '1rem 0' }}>
        {items.map(item => (
          <div key={item.key} onClick={() => item.key === 'dashboard' && navigate('/investigator/dashboard')}
            style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1.5rem', cursor: 'pointer', backgroundColor: active === item.key ? 'rgba(201,168,76,0.15)' : 'transparent', borderLeft: active === item.key ? '3px solid #C9A84C' : '3px solid transparent' }}>
            <span style={{ color: active === item.key ? '#C9A84C' : 'rgba(255,255,255,0.5)', width: 20, textAlign: 'center' }}>{item.icon}</span>
            <span style={{ color: active === item.key ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: active === item.key ? 600 : 400, flex: 1 }}>{item.label}</span>
          </div>
        ))}
      </nav>
      <div style={{ padding: '1rem 1.5rem', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
        <div onClick={() => { logout(); window.location.href = '/login' }} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
          <span style={{ color: 'rgba(255,255,255,0.4)' }}>↩</span>
          <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>
        </div>
      </div>
    </div>
  )
}

function ScoreGauge({ score }) {
  const color = score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A'
  const circumference = 2 * Math.PI * 50
  const offset = circumference - (score / 100) * circumference
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <svg width={120} height={120} viewBox="0 0 120 120">
          <circle cx={60} cy={60} r={50} fill="none" stroke="#F3F4F6" strokeWidth={9} />
          <circle cx={60} cy={60} r={50} fill="none" stroke={color} strokeWidth={9}
            strokeDasharray={circumference} strokeDashoffset={offset}
            strokeLinecap="round" transform="rotate(-90 60 60)" />
        </svg>
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', textAlign: 'center' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: '0.6rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
    </div>
  )
}

export default function ClaimReview() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthStore()

  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
<<<<<<< HEAD
  const [decision, setDecision] = useState('')
  const [comment, setComment] = useState('')
=======
  // FIX 1 — renamed to match backend field names
  const [outcome, setOutcome] = useState('')  // 'APPROVED' | 'REJECTED'
  const [notes, setNotes] = useState('')
>>>>>>> a259412 (frontend v2 not completed)
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)

  useEffect(() => {
    api.get(`/claims/${id}`)
<<<<<<< HEAD
      .then(res => setClaim(res.data))
      .catch(err => setError('Dossier introuvable'))
=======
      .then(res => setClaim(res.data?.data || res.data))
      .catch(() => setError('Dossier introuvable'))
>>>>>>> a259412 (frontend v2 not completed)
      .finally(() => setLoading(false))
  }, [id])

  const handleDecision = async () => {
<<<<<<< HEAD
    if (!decision || !comment.trim()) return
    setSubmitting(true)
    try {
      await api.patch(`/claims/${id}/decide`, {
        decision,
        note: comment
      })
=======
    if (!outcome || !notes.trim()) return
    setSubmitting(true)
    try {
      // FIX 1 — correct field names
      await api.patch(`/claims/${id}/decide`, { outcome, notes })
>>>>>>> a259412 (frontend v2 not completed)
      setDone(true)
    } catch (err) {
      setError(err.response?.data?.message || 'Erreur lors de la soumission')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="flagged" />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
      </div>
    </div>
  )

  if (error || !claim) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="flagged" />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>404</div>
          <div style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>{error}</div>
          <button onClick={() => navigate('/investigator/dashboard')} style={{ padding: '0.6rem 1.2rem', background: '#0F2347', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Retour</button>
        </div>
      </div>
    </div>
  )

<<<<<<< HEAD
=======
  // FIX 2 — client name from firstName + lastName
  const clientName = `${claim.client?.firstName || ''} ${claim.client?.lastName || ''}`.trim() || 'Client'
  // FIX 3 — equipment is object
  const equipmentName = claim.equipment?.name || '-'
  // FIX 4 — scores from analysis
  const analysis = claim.analysis || {}
  const finalScore = analysis.finalScore ?? 50
  const anomalyScore = analysis.anomalyScore ?? 50
  const classificationScore = analysis.classificationScore ?? 50
  const nlpScore = analysis.nlpScore ?? 50
  const visionScore = analysis.visionScore ?? 50
  // FIX 5 — claimedAmount
  const claimedAmount = claim.claimedAmount

  const score = Math.round(finalScore)
  // FIX 2 — investigator name
  const investigatorName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'

>>>>>>> a259412 (frontend v2 not completed)
  if (done) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="flagged" />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', backgroundColor: 'white', borderRadius: 16, padding: '3rem', border: '1px solid #EEF0F6', maxWidth: 480 }}>
<<<<<<< HEAD
          <div style={{ width: 64, height: 64, borderRadius: '50%', backgroundColor: decision === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `2px solid ${decision === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', margin: '0 auto 1.5rem' }}>
            {decision === 'APPROVED' ? '✓' : '✕'}
          </div>
          <h2 style={{ color: '#0F2347', fontSize: '1.4rem', fontWeight: 400, marginBottom: '0.5rem' }}>
            Dossier <strong>{decision === 'APPROVED' ? 'Approuve' : 'Rejete'}</strong>
          </h2>
          <p style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.88rem', marginBottom: '0.5rem' }}>
            {claim.reference} — {claim.client?.fullName || 'Client'}
          </p>
          <p style={{ color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '2rem', fontStyle: 'italic' }}>
            "{comment}"
          </p>
          <div style={{ backgroundColor: '#F7F8FC', borderRadius: 8, padding: '0.75rem', marginBottom: '2rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            Decision enregistree par <strong style={{ color: '#0F2347' }}>{user?.fullName}</strong> — {new Date().toLocaleString('fr-FR')}
=======
          <div style={{ width: 64, height: 64, borderRadius: '50%', backgroundColor: outcome === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `2px solid ${outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', margin: '0 auto 1.5rem' }}>
            {outcome === 'APPROVED' ? '✓' : '✕'}
          </div>
          <h2 style={{ color: '#0F2347', fontSize: '1.4rem', fontWeight: 400, marginBottom: '0.5rem' }}>
            Dossier <strong>{outcome === 'APPROVED' ? 'Approuve' : 'Rejete'}</strong>
          </h2>
          <p style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.88rem', marginBottom: '0.5rem' }}>
            {claim.reference} — {clientName}
          </p>
          <p style={{ color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '2rem', fontStyle: 'italic' }}>
            "{notes}"
          </p>
          <div style={{ backgroundColor: '#F7F8FC', borderRadius: 8, padding: '0.75rem', marginBottom: '2rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            Decision enregistree par <strong style={{ color: '#0F2347' }}>{investigatorName}</strong> — {new Date().toLocaleString('fr-FR')}
>>>>>>> a259412 (frontend v2 not completed)
          </div>
          <button onClick={() => navigate('/investigator/dashboard')}
            style={{ padding: '0.75rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
            Retour au dashboard
          </button>
        </div>
      </div>
    </div>
  )

<<<<<<< HEAD
  const score = Math.round(claim.finalScore || 50)

=======
>>>>>>> a259412 (frontend v2 not completed)
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="flagged" />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

<<<<<<< HEAD
        {/* Header */}
=======
>>>>>>> a259412 (frontend v2 not completed)
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <button onClick={() => navigate('/investigator/dashboard')}
              style={{ background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '0.5rem' }}>
              ← Retour au dashboard
            </button>
            <h1 style={{ fontSize: '1.75rem', color: '#0F2347', fontWeight: 400 }}>
              Revision dossier <strong>{claim.reference}</strong>
            </h1>
<<<<<<< HEAD
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {claim.client?.fullName} — {claim.equipment}
=======
            {/* FIX 2+3 */}
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {clientName} — {equipmentName}
>>>>>>> a259412 (frontend v2 not completed)
            </p>
          </div>
          <div style={{ backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, padding: '0.5rem 1rem', fontSize: '0.82rem', color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>
            Revision humaine requise
          </div>
        </div>

        {error && (
          <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 6, padding: '0.7rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.5rem' }}>
<<<<<<< HEAD

          {/* Gauche */}
          <div>
            {/* Infos */}
=======
          {/* Left */}
          <div>
>>>>>>> a259412 (frontend v2 not completed)
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Informations du dossier</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                {[
<<<<<<< HEAD
                  ['Client',          claim.client?.fullName || '-'],
                  ['Montant reclame', `${claim.amount?.toLocaleString('fr-FR')} DA`],
                  ['Equipement',      claim.equipment],
                  ['Date incident',   new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Lieu',            claim.location || 'Non specifie'],
                  ['Indicateur IA',   claim.fraudIndicator || 'UNKNOWN'],
=======
                  /* FIX 2 */['Client', clientName],
                  /* FIX 5 */['Montant reclame', claimedAmount ? `${claimedAmount.toLocaleString('fr-FR')} DA` : '-'],
                  /* FIX 3 */['Equipement', equipmentName],
                  ['Date incident', new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Lieu', claim.location || 'Non specifie'],
                  /* FIX 4 */['Classe fraude', analysis.fraudClass || 'N/A'],
>>>>>>> a259412 (frontend v2 not completed)
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                    <div style={{ fontSize: '0.88rem', color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Description</div>
                <div style={{ fontSize: '0.88rem', color: '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.65 }}>{claim.description}</div>
              </div>
            </div>

<<<<<<< HEAD
            {/* Scores IA */}
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Detail des 4 modeles IA</h2>
              {[
                ['Modele 1 — Anomalie capteurs (35%)',  claim.anomalyScore],
                ['Modele 2 — Classification (25%)',     claim.classificationScore],
                ['Modele 3 — Analyse NLP (20%)',        claim.nlpScore],
                ['Modele 4 — Vision photos (20%)',      claim.visionScore],
              ].map(([title, sc]) => {
                const s = Math.round(sc || 50)
=======
            {/* AI Scores — FIX 4 */}
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Detail des 4 modeles IA</h2>
              {[
                ['Modele 1 — Anomalie capteurs (35%)', anomalyScore],
                ['Modele 2 — Classification (25%)', classificationScore],
                ['Modele 3 — Analyse NLP (20%)', nlpScore],
                ['Modele 4 — Vision photos (20%)', visionScore],
              ].map(([title, sc]) => {
                const s = Math.round(sc)
>>>>>>> a259412 (frontend v2 not completed)
                const color = s > 70 ? '#C0392B' : s > 30 ? '#F39C12' : '#1A7A4A'
                return (
                  <div key={title} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', padding: '0.85rem', border: '1px solid #EEF0F6', borderRadius: 8, marginBottom: '0.6rem' }}>
                    <div style={{ textAlign: 'center', minWidth: 48 }}>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s}</div>
                      <div style={{ fontSize: '0.6rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 5, backgroundColor: '#F3F4F6', borderRadius: 3, overflow: 'hidden', marginBottom: '0.4rem' }}>
                        <div style={{ height: '100%', width: `${s}%`, backgroundColor: color, borderRadius: 3 }} />
                      </div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{title}</div>
                      <div style={{ fontSize: '0.75rem', color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, marginTop: 2 }}>
                        {s > 70 ? 'Score eleve — suspect' : s > 30 ? 'Zone grise' : 'Score normal'}
                      </div>
                    </div>
                  </div>
                )
              })}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', backgroundColor: '#F7F8FC', borderRadius: 8, marginTop: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combine</span>
                <span style={{ fontSize: '1.2rem', fontWeight: 700, color: score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  {score} / 100
                </span>
              </div>
            </div>
          </div>

<<<<<<< HEAD
          {/* Droite */}
          <div>
            {/* Score */}
=======
          {/* Right */}
          <div>
>>>>>>> a259412 (frontend v2 not completed)
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem', textAlign: 'center' }}>Score IA</h2>
              <ScoreGauge score={score} />
              <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>ZONE GRISE</div>
                <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>Revision humaine obligatoire</div>
              </div>
<<<<<<< HEAD
              {claim.fraudIndicator && (
                <div style={{ marginTop: '0.75rem', padding: '0.65rem', backgroundColor: '#F7F8FC', borderRadius: 8, textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.fraudIndicator}</div>
                  <div style={{ fontSize: '0.68rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Indicateur de fraude</div>
                </div>
              )}
            </div>

            {/* Decision */}
=======
            </div>

            {/* Decision panel — FIX 1 */}
>>>>>>> a259412 (frontend v2 not completed)
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', position: 'sticky', top: '2rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Votre decision</h2>
              <p style={{ color: '#9CA3AF', fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>
                Votre decision est definitive et sera notifiee au client
              </p>

              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                {[
                  { key: 'APPROVED', icon: '✓', label: 'Approuver', desc: 'Sinistre legitime', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA' },
<<<<<<< HEAD
                  { key: 'REJECTED', icon: '✕', label: 'Rejeter',   desc: 'Fraude probable',  color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE' },
                ].map(r => (
                  <div key={r.key} onClick={() => setDecision(r.key)}
                    style={{ flex: 1, padding: '1rem', border: `2px solid ${decision === r.key ? r.color : '#E5E7EB'}`, borderRadius: 10, cursor: 'pointer', textAlign: 'center', backgroundColor: decision === r.key ? r.bg : '#F9FAFB', transition: 'all 0.2s' }}>
                    <div style={{ fontSize: '1.4rem', marginBottom: '0.3rem', color: decision === r.key ? r.color : '#9CA3AF' }}>{r.icon}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: decision === r.key ? r.color : '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r.label}</div>
=======
                  { key: 'REJECTED', icon: '✕', label: 'Rejeter', desc: 'Fraude probable', color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE' },
                ].map(r => (
                  <div key={r.key} onClick={() => setOutcome(r.key)}
                    style={{ flex: 1, padding: '1rem', border: `2px solid ${outcome === r.key ? r.color : '#E5E7EB'}`, borderRadius: 10, cursor: 'pointer', textAlign: 'center', backgroundColor: outcome === r.key ? r.bg : '#F9FAFB', transition: 'all 0.2s' }}>
                    <div style={{ fontSize: '1.4rem', marginBottom: '0.3rem', color: outcome === r.key ? r.color : '#9CA3AF' }}>{r.icon}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: outcome === r.key ? r.color : '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r.label}</div>
>>>>>>> a259412 (frontend v2 not completed)
                    <div style={{ fontSize: '0.7rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{r.desc}</div>
                  </div>
                ))}
              </div>

              <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#4B5563', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
<<<<<<< HEAD
                Commentaire obligatoire
              </label>
              <textarea rows={4} placeholder="Justifiez votre decision en detail..."
                value={comment} onChange={e => setComment(e.target.value)}
                style={{ width: '100%', padding: '0.72rem', border: `1.5px solid ${!comment && decision ? '#C0392B' : '#E5E7EB'}`, borderRadius: 6, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box', resize: 'vertical' }} />

              <button onClick={handleDecision} disabled={!decision || !comment.trim() || submitting}
                style={{ width: '100%', marginTop: '1rem', padding: '0.85rem',
                  background: !decision || !comment.trim() ? '#E5E7EB'
                    : decision === 'APPROVED' ? 'linear-gradient(135deg, #1A7A4A, #27AE60)'
                    : 'linear-gradient(135deg, #C0392B, #E74C3C)',
                  color: !decision || !comment.trim() ? '#9CA3AF' : 'white',
                  border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
                  fontWeight: 600, cursor: !decision || !comment.trim() || submitting ? 'not-allowed' : 'pointer',
                  letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {submitting ? 'Enregistrement...'
                  : !decision ? 'Choisissez une decision'
                  : decision === 'APPROVED' ? 'Confirmer — Approuver'
                  : 'Confirmer — Rejeter'}
=======
                Commentaire obligatoire (min. 10 caracteres)
              </label>
              <textarea rows={4} placeholder="Justifiez votre decision en detail..."
                value={notes} onChange={e => setNotes(e.target.value)}
                style={{ width: '100%', padding: '0.72rem', border: `1.5px solid ${!notes && outcome ? '#C0392B' : '#E5E7EB'}`, borderRadius: 6, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box', resize: 'vertical' }} />

              <button onClick={handleDecision}
                disabled={!outcome || notes.trim().length < 10 || submitting}
                style={{
                  width: '100%', marginTop: '1rem', padding: '0.85rem',
                  background: !outcome || notes.trim().length < 10
                    ? '#E5E7EB'
                    : outcome === 'APPROVED'
                      ? 'linear-gradient(135deg, #1A7A4A, #27AE60)'
                      : 'linear-gradient(135deg, #C0392B, #E74C3C)',
                  color: !outcome || notes.trim().length < 10 ? '#9CA3AF' : 'white',
                  border: 'none', borderRadius: 8, fontSize: '0.86rem',
                  fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600,
                  cursor: !outcome || notes.trim().length < 10 || submitting ? 'not-allowed' : 'pointer',
                  letterSpacing: '0.05em', textTransform: 'uppercase',
                }}>
                {submitting
                  ? 'Enregistrement...'
                  : !outcome
                    ? 'Choisissez une decision'
                    : outcome === 'APPROVED'
                      ? 'Confirmer — Approuver'
                      : 'Confirmer — Rejeter'}
>>>>>>> a259412 (frontend v2 not completed)
              </button>

              <p style={{ fontSize: '0.72rem', color: '#9CA3AF', textAlign: 'center', marginTop: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                Cette action est irreversible
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}