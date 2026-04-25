import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { exportClaimPDF } from '../../utils/pdfExport'

<<<<<<< HEAD
const STATUS_CONFIG = {
  APPROVED:     { label: 'Approuve',         bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED:     { label: 'Rejete',           bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING:      { label: 'En attente',       bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING:    { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
=======
/**
 * ClaimDetail
 *
 * FIX 1 — All AI scores are nested under `claim.analysis`:
 *          claim.analysis?.finalScore, claim.analysis?.anomalyScore, etc.
 *
 * FIX 2 — Decision data lives in `claim.decision`:
 *          claim.decision?.notes, claim.decision?.outcome, claim.decision?.decidedAt,
 *          claim.decision?.investigator (object with firstName/lastName).
 *
 * FIX 3 — `claim.equipment` is an object {name, type, ...}, not a string.
 *
 * FIX 4 — `claim.claimedAmount` not `claim.amount`.
 *
 * FIX 5 — `fraudIndicator` / `preIncidentAnomaly` don't exist in the backend
 *          schema. Removed references; score thresholds used instead.
 */

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuve', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED: { label: 'Rejete', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
>>>>>>> a259412 (frontend v2 not completed)
  HUMAN_REVIEW: { label: 'Revision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
}

function Sidebar({ active }) {
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()
  const items = [
    { key: 'dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: 'new-claim', label: 'Nouveau sinistre', icon: '+' },
<<<<<<< HEAD
    { key: 'claims',    label: 'Mes sinistres',    icon: '≡' },
    { key: 'profile',   label: 'Mon profil',       icon: '👤' },
=======
    { key: 'claims', label: 'Mes sinistres', icon: '≡' },
    { key: 'profile', label: 'Mon profil', icon: '👤' },
>>>>>>> a259412 (frontend v2 not completed)
  ]
  return (
    <div style={{ width: 240, minHeight: '100vh', backgroundColor: '#0F2347', display: 'flex', flexDirection: 'column', position: 'fixed', left: 0, top: 0, zIndex: 100 }}>
      <div style={{ padding: '1.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347' }}>F</div>
          <div>
            <div style={{ color: 'white', fontWeight: 700, fontSize: '0.95rem' }}>FraudGuard AI</div>
            <div style={{ color: '#C9A84C', fontSize: '0.62rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Espace Client</div>
          </div>
        </div>
      </div>
      <div style={{ padding: '1rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ width: 38, height: 38, borderRadius: '50%', backgroundColor: '#C9A84C', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, marginBottom: '0.5rem' }}>
<<<<<<< HEAD
          {user?.fullName?.[0] || 'U'}
        </div>
        <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{user?.fullName || 'Utilisateur'}</div>
=======
          {user?.firstName?.[0] || 'U'}
        </div>
        <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>
          {`${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Utilisateur'}
        </div>
>>>>>>> a259412 (frontend v2 not completed)
        <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{user?.company || 'Client'}</div>
      </div>
      <nav style={{ flex: 1, padding: '1rem 0' }}>
        {items.map(item => (
          <div key={item.key}
<<<<<<< HEAD
            onClick={() => item.key === 'new-claim' ? navigate('/client/new-claim') : item.key === 'dashboard' ? navigate('/client/dashboard') : null}
=======
            onClick={() => {
              if (item.key === 'new-claim') navigate('/client/new-claim')
              if (item.key === 'dashboard') navigate('/client/dashboard')
              if (item.key === 'claims') navigate('/client/claims')
              if (item.key === 'profile') navigate('/client/profile')
            }}
>>>>>>> a259412 (frontend v2 not completed)
            style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1.5rem', cursor: 'pointer', backgroundColor: active === item.key ? 'rgba(201,168,76,0.15)' : 'transparent', borderLeft: active === item.key ? '3px solid #C9A84C' : '3px solid transparent' }}>
            <span style={{ color: active === item.key ? '#C9A84C' : 'rgba(255,255,255,0.5)', width: 20, textAlign: 'center' }}>{item.icon}</span>
            <span style={{ color: active === item.key ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: active === item.key ? 600 : 400 }}>{item.label}</span>
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
  const circumference = 2 * Math.PI * 54
  const offset = circumference - (score / 100) * circumference
  return (
    <div style={{ textAlign: 'center', padding: '1.5rem' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <svg width={140} height={140} viewBox="0 0 140 140">
          <circle cx={70} cy={70} r={54} fill="none" stroke="#F3F4F6" strokeWidth={10} />
          <circle cx={70} cy={70} r={54} fill="none" stroke={color} strokeWidth={10}
            strokeDasharray={circumference} strokeDashoffset={offset}
            strokeLinecap="round" transform="rotate(-90 70 70)" />
        </svg>
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: '0.65rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
      <div style={{ marginTop: '0.75rem' }}>
        <div style={{ fontSize: '0.88rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          {score > 70 ? 'Fraude tres probable' : score > 30 ? 'Zone grise' : 'Sinistre credible'}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>Score global de fraude</div>
      </div>
    </div>
  )
}

function AIModelCard({ title, score, weight }) {
  const color = score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A'
  const label = score > 70 ? 'Score eleve — suspect' : score > 30 ? 'Zone grise' : 'Score normal'
  return (
    <div style={{ backgroundColor: 'white', border: '1px solid #EEF0F6', borderRadius: 10, padding: '1rem', marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{title}</div>
          <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Poids : {weight}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{Math.round(score)}</div>
          <div style={{ fontSize: '0.65rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
      <div style={{ height: 6, backgroundColor: '#F3F4F6', borderRadius: 3, overflow: 'hidden', marginBottom: '0.5rem' }}>
        <div style={{ height: '100%', width: `${score}%`, backgroundColor: color, borderRadius: 3 }} />
      </div>
      <div style={{ fontSize: '0.78rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{label}</div>
    </div>
  )
}

export default function ClaimDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [exporting, setExporting] = useState(false)

  useEffect(() => {
    api.get(`/claims/${id}`)
      .then(res => setClaim(res.data?.data || res.data))
      .catch(err => {
        if (err.response?.status === 404) setError('Sinistre introuvable')
        else setError('Erreur lors du chargement')
      })
      .finally(() => setLoading(false))
  }, [id])

<<<<<<< HEAD
=======
  // Poll while still processing
>>>>>>> a259412 (frontend v2 not completed)
  useEffect(() => {
    if (!claim) return
    if (claim.status !== 'ANALYZING' && claim.status !== 'PENDING') return
    const interval = setInterval(() => {
      api.get(`/claims/${id}`)
        .then(res => setClaim(res.data?.data || res.data))
<<<<<<< HEAD
        .catch(() => {})
=======
        .catch(() => { })
>>>>>>> a259412 (frontend v2 not completed)
    }, 5000)
    return () => clearInterval(interval)
  }, [claim, id])

  const handleExportPDF = async () => {
    setExporting(true)
    try {
      exportClaimPDF(claim, user)
    } catch (err) {
      console.error('Erreur export PDF:', err)
    } finally {
      setExporting(false)
    }
  }

  if (loading) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="claims" />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
      </div>
    </div>
  )

  if (error || !claim) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="claims" />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>404</div>
          <div style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>{error || 'Sinistre introuvable'}</div>
          <button onClick={() => navigate('/client/dashboard')}
            style={{ padding: '0.6rem 1.2rem', background: '#0F2347', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            Retour
          </button>
        </div>
      </div>
    </div>
  )

  const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']

<<<<<<< HEAD
=======
  // FIX 1 — all AI data is nested inside claim.analysis
  const analysis = claim.analysis || null
  const finalScore = analysis?.finalScore ?? null
  const anomalyScore = analysis?.anomalyScore ?? null
  const classificationScore = analysis?.classificationScore ?? null
  const nlpScore = analysis?.nlpScore ?? null
  const visionScore = analysis?.visionScore ?? null
  const fraudClass = analysis?.fraudClass ?? null

  // FIX 2 — decision nested under claim.decision
  const decision = claim.decision || null
  const investigatorNotes = decision?.notes ?? null
  const decidedAt = decision?.createdAt ?? null
  const investigatorName = decision
    ? `${decision.investigator?.firstName || ''} ${decision.investigator?.lastName || ''}`.trim()
    : null

  // FIX 3 — equipment is object
  const equipmentName = claim.equipment?.name || '-'
  const equipmentType = claim.equipment?.type || '-'

  // FIX 4 — claimedAmount
  const claimedAmount = claim.claimedAmount

>>>>>>> a259412 (frontend v2 not completed)
  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <Sidebar active="claims" />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <button onClick={() => navigate('/client/dashboard')}
              style={{ background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '0.5rem' }}>
              ← Retour au dashboard
            </button>
            <h1 style={{ fontSize: '1.75rem', color: '#0F2347', fontWeight: 400 }}>
              Sinistre <strong>{claim.reference}</strong>
            </h1>
<<<<<<< HEAD
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {claim.equipment?.name || claim.equipment} — {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
=======
            {/* FIX 3 */}
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {equipmentName} — {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
>>>>>>> a259412 (frontend v2 not completed)
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button onClick={handleExportPDF} disabled={exporting}
              style={{ padding: '0.5rem 1.1rem', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', color: '#0F2347', border: 'none', borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 700, cursor: exporting ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem', opacity: exporting ? 0.7 : 1, boxShadow: '0 2px 8px rgba(201,168,76,0.3)', transition: 'transform 0.15s' }}
              onMouseEnter={e => { if (!exporting) e.currentTarget.style.transform = 'translateY(-1px)' }}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
              📄 {exporting ? 'Export...' : 'Exporter PDF'}
            </button>
            <span style={{ padding: '0.4rem 1rem', borderRadius: 20, fontSize: '0.82rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sc.bg, color: sc.color, border: `1px solid ${sc.border}` }}>
              {sc.label}
            </span>
          </div>
        </div>

        {(claim.status === 'ANALYZING' || claim.status === 'PENDING') && (
          <div style={{ backgroundColor: '#EBF5FB', border: '1px solid #AED6F1', borderRadius: 10, padding: '1rem 1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ fontSize: '1.5rem' }}>⏳</div>
            <div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#1A5276', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Analyse IA en cours...</div>
              <div style={{ fontSize: '0.8rem', color: '#5D9CEC', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Cette page se rafraichit automatiquement toutes les 5 secondes</div>
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>
          <div>
<<<<<<< HEAD
=======
            {/* Claim info */}
>>>>>>> a259412 (frontend v2 not completed)
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Informations du sinistre</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                {[
<<<<<<< HEAD
                  ['Equipement',      claim.equipment?.name || claim.equipment],
                  ['Montant reclame', `${claim.amount?.toLocaleString('fr-FR')} DA`],
                  ['Date incident',   new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Lieu',            claim.location || 'Non specifie'],
=======
                  ['Equipement', equipmentName],
                  ['Type', equipmentType],
                  /* FIX 4 */
                  ['Montant reclame', claimedAmount ? `${claimedAmount.toLocaleString('fr-FR')} DA` : '-'],
                  ['Date incident', new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
>>>>>>> a259412 (frontend v2 not completed)
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                    <div style={{ fontSize: '0.9rem', color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Description</div>
                <div style={{ fontSize: '0.88rem', color: '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.6 }}>{claim.description}</div>
              </div>
            </div>

<<<<<<< HEAD
            {claim.finalScore !== null && claim.finalScore !== undefined && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Analyse par les 4 modeles IA</h2>
                {claim.anomalyScore !== null && <AIModelCard title="Modele 1 — Anomalie capteurs" score={claim.anomalyScore} weight="35%" />}
                {claim.classificationScore !== null && <AIModelCard title="Modele 2 — Classification panne" score={claim.classificationScore} weight="25%" />}
                {claim.nlpScore !== null && <AIModelCard title="Modele 3 — Analyse rapport NLP" score={claim.nlpScore} weight="20%" />}
                {claim.visionScore !== null && <AIModelCard title="Modele 4 — Verification photos" score={claim.visionScore} weight="20%" />}
                <div style={{ backgroundColor: '#F7F8FC', border: '1px solid #EEF0F6', borderRadius: 8, padding: '0.75rem 1rem', marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combine</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: 700, color: claim.finalScore > 70 ? '#C0392B' : claim.finalScore > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    {Math.round(claim.finalScore)} / 100
                  </span>
                </div>
              </div>
            )}

            {claim.investigatorNote && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>Decision de l'investigateur</h2>
                <div style={{ backgroundColor: claim.status === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `1px solid ${claim.status === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, borderRadius: 8, padding: '0.75rem 1rem', fontSize: '0.88rem', color: claim.status === 'APPROVED' ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontStyle: 'italic' }}>
                  "{claim.investigatorNote}"
                </div>
                {claim.decidedAt && (
                  <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.5rem' }}>
                    Decision prise le {new Date(claim.decidedAt).toLocaleString('fr-FR')}
=======
            {/* AI analysis — FIX 1 */}
            {finalScore !== null && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Analyse par les 4 modeles IA</h2>
                {anomalyScore !== null && <AIModelCard title="Modele 1 — Anomalie capteurs" score={anomalyScore} weight="35%" />}
                {classificationScore !== null && <AIModelCard title="Modele 2 — Classification panne" score={classificationScore} weight="25%" />}
                {nlpScore !== null && <AIModelCard title="Modele 3 — Analyse rapport NLP" score={nlpScore} weight="20%" />}
                {visionScore !== null && <AIModelCard title="Modele 4 — Verification photos" score={visionScore} weight="20%" />}
                <div style={{ backgroundColor: '#F7F8FC', border: '1px solid #EEF0F6', borderRadius: 8, padding: '0.75rem 1rem', marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combine</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: 700, color: finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    {Math.round(finalScore)} / 100
                  </span>
                </div>
                {fraudClass && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    Classe de fraude : <strong style={{ color: '#0F2347' }}>{fraudClass}</strong>
                  </div>
                )}
              </div>
            )}

            {/* Decision — FIX 2 */}
            {investigatorNotes && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>
                  Decision de l'investigateur
                  {investigatorName && <span style={{ fontWeight: 400, color: '#9CA3AF', fontSize: '0.82rem', marginLeft: '0.5rem' }}>— {investigatorName}</span>}
                </h2>
                <div style={{ backgroundColor: claim.status === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `1px solid ${claim.status === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, borderRadius: 8, padding: '0.75rem 1rem', fontSize: '0.88rem', color: claim.status === 'APPROVED' ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontStyle: 'italic' }}>
                  "{investigatorNotes}"
                </div>
                {decidedAt && (
                  <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.5rem' }}>
                    Decision prise le {new Date(decidedAt).toLocaleString('fr-FR')}
>>>>>>> a259412 (frontend v2 not completed)
                  </div>
                )}
              </div>
            )}
          </div>

<<<<<<< HEAD
          <div>
            {claim.finalScore !== null && claim.finalScore !== undefined ? (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', marginBottom: '1.5rem' }}>
                <ScoreGauge score={Math.round(claim.finalScore)} />
                <div style={{ padding: '0 1.5rem 1.5rem' }}>
                  {claim.fraudIndicator && (
                    <div style={{ backgroundColor: claim.finalScore > 70 ? '#FDF2F2' : claim.finalScore > 30 ? '#FEF9E7' : '#F0FAF4', border: `1px solid ${claim.finalScore > 70 ? '#EBCECE' : claim.finalScore > 30 ? '#F7DC6F' : '#B8E4CA'}`, borderRadius: 8, padding: '0.75rem', textAlign: 'center' }}>
                      <div style={{ fontSize: '0.78rem', fontWeight: 600, color: claim.finalScore > 70 ? '#C0392B' : claim.finalScore > 30 ? '#7D6608' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.25rem' }}>
                        {claim.fraudIndicator}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Indicateur de fraude</div>
                    </div>
                  )}
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#F7F8FC', borderRadius: 8 }}>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>Precurseurs avant incident</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: claim.preIncidentAnomaly ? '#1A7A4A' : '#C0392B' }} />
                      <span style={{ fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', color: claim.preIncidentAnomaly ? '#1A7A4A' : '#C0392B', fontWeight: 600 }}>
                        {claim.preIncidentAnomaly ? 'Anomalies detectees' : 'Aucun precurseur'}
                      </span>
                    </div>
                  </div>
                </div>
=======
          {/* Right column — score gauge */}
          <div>
            {finalScore !== null ? (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', marginBottom: '1.5rem' }}>
                <ScoreGauge score={Math.round(finalScore)} />
>>>>>>> a259412 (frontend v2 not completed)
              </div>
            ) : (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '2rem', marginBottom: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⏳</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>Analyse en cours</div>
                <div style={{ fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Le score IA sera disponible dans quelques secondes</div>
              </div>
            )}

<<<<<<< HEAD
=======
            {/* Timeline */}
>>>>>>> a259412 (frontend v2 not completed)
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Historique</h2>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {[
<<<<<<< HEAD
                  { label: 'Sinistre soumis',   date: claim.createdAt,  done: true },
                  { label: 'Analyse IA lancee', date: claim.createdAt,  done: true },
                  { label: 'Analyse terminee',  date: claim.updatedAt,  done: claim.finalScore !== null },
                  { label: 'Decision finale',   date: claim.decidedAt,  done: !!claim.decidedAt },
=======
                  { label: 'Sinistre soumis', date: claim.createdAt, done: true },
                  { label: 'Analyse IA lancee', date: claim.createdAt, done: true },
                  { label: 'Analyse terminee', date: claim.updatedAt, done: finalScore !== null },
                  { label: 'Decision finale', date: decidedAt, done: !!decidedAt },
>>>>>>> a259412 (frontend v2 not completed)
                ].map((t, i, arr) => (
                  <div key={i} style={{ display: 'flex', gap: '0.75rem', paddingBottom: i < arr.length - 1 ? '1rem' : 0 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: t.done ? '#1A7A4A' : '#E5E7EB', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        {t.done && <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'white' }} />}
                      </div>
                      {i < arr.length - 1 && <div style={{ width: 2, flex: 1, backgroundColor: t.done ? '#B8E4CA' : '#E5E7EB', marginTop: 4 }} />}
                    </div>
                    <div style={{ paddingBottom: i < arr.length - 1 ? '0.5rem' : 0 }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{t.label}</div>
                      {t.date && <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(t.date).toLocaleString('fr-FR')}</div>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}