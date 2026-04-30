import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import { InvestigatorSidebar } from '../../components/layout/InvestigatorLayout'
import NotificationBell from '../../components/ui/NotificationBell'
import { useNotifications } from '../../hooks/useNotifications'

/**
 * FlaggedClaims — Feature 10
 * Full paginated work queue of all HUMAN_REVIEW claims sorted by fraud score.
 * Route: /investigator/flagged
 */

const PRIORITY_CONFIG = {
    HIGH: { label: 'Urgent', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
    MEDIUM: { label: 'Moyen', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
    LOW: { label: 'Faible', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
}

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

export default function FlaggedClaims() {
    const navigate = useNavigate()
    const [dark, toggleDark] = useDarkMode()
    const [claims, setClaims] = useState([])
    const [loading, setLoading] = useState(true)
    const [filter, setFilter] = useState('ALL')
    const [search, setSearch] = useState('')
    const [sortBy, setSortBy] = useState('score')
    const [sortDir, setSortDir] = useState('desc')

    const { unreadCount } = useNotifications()

    const fetchClaims = () =>
        api.get('/claims/flagged?limit=100')
            .then(res => setClaims(extractArray(res.data)))
            .catch(console.error)
            .finally(() => setLoading(false))

    useEffect(() => { fetchClaims() }, [])
    useEffect(() => {
        const id = setInterval(fetchClaims, 15000)
        return () => clearInterval(id)
    }, [])

    const handleSort = (field) => {
        if (sortBy === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
        else { setSortBy(field); setSortDir('desc') }
    }
    const sortIcon = (f) => sortBy !== f ? ' ↕' : sortDir === 'asc' ? ' ↑' : ' ↓'

    const filtered = claims
        .filter(c => {
            const score = c.analysis?.finalScore ?? 50
            const priority = getPriority(score)
            const matchFilter = filter === 'ALL' || priority === filter
            const q = search.toLowerCase()
            const matchSearch = !q
                || c.reference?.toLowerCase().includes(q)
                || clientName(c.client).toLowerCase().includes(q)
                || (c.equipment?.name || '').toLowerCase().includes(q)
            return matchFilter && matchSearch
        })
        .sort((a, b) => {
            let va, vb
            if (sortBy === 'score') { va = a.analysis?.finalScore ?? 0; vb = b.analysis?.finalScore ?? 0 }
            else if (sortBy === 'date') { va = new Date(a.createdAt).getTime(); vb = new Date(b.createdAt).getTime() }
            else if (sortBy === 'amount') { va = a.claimedAmount ?? 0; vb = b.claimedAmount ?? 0 }
            return sortDir === 'asc' ? va - vb : vb - va
        })

    const pageBg = dark ? '#0D1626' : '#F7F8FC'
    const cardBg = dark ? '#111C30' : 'white'
    const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
    const textMain = dark ? 'white' : '#0F2347'
    const textSub = dark ? '#5A7A9A' : '#9CA3AF'
    const textBody = dark ? '#C8D8E8' : '#4B5563'
    const rowHover = dark ? '#172338' : '#F9FAFB'

    const thStyle = (f) => ({
        fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '0.06em', color: sortBy === f ? '#C9A84C' : textSub,
        fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', userSelect: 'none',
    })

    return (
        <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
            <InvestigatorSidebar dark={dark} badgeCount={unreadCount} />

            <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
                    <div>
                        <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>File d'attente</p>
                        <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Dossiers <strong>à traiter</strong></h1>
                        <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                            {filtered.length} dossier{filtered.length > 1 ? 's' : ''} en revision humaine
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

                {/* Table */}
                <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>

                    {/* Toolbar */}
                    <div style={{ padding: '1.25rem 1.5rem', borderBottom: `1px solid ${cardBorder}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                        <div>
                            <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>HUMAN_REVIEW — trie par score decroissant</h2>
                            <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: '0.15rem 0 0' }}>{claims.length} dossier{claims.length > 1 ? 's' : ''} au total</p>
                        </div>
                        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                            <input placeholder="Rechercher..." value={search} onChange={e => setSearch(e.target.value)}
                                style={{ padding: '0.5rem 0.9rem', border: `1.5px solid ${cardBorder}`, borderRadius: 6, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', width: 200, backgroundColor: cardBg, color: textMain }} />
                            {[['ALL', 'Tous'], ['HIGH', 'Urgent'], ['MEDIUM', 'Moyen'], ['LOW', 'Faible']].map(([f, l]) => (
                                <button key={f} onClick={() => setFilter(f)}
                                    style={{ padding: '0.4rem 0.85rem', border: `1.5px solid ${filter === f ? '#0F2347' : cardBorder}`, borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: filter === f ? '#0F2347' : cardBg, color: filter === f ? 'white' : textSub, fontWeight: filter === f ? 600 : 400 }}>{l}</button>
                            ))}
                        </div>
                    </div>

                    {/* Column headers */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.4fr 1.8fr 0.9fr 1.1fr 0.9fr 0.9fr 90px', padding: '0.75rem 1.5rem', backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderBottom: `1px solid ${cardBorder}` }}>
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
                                {claims.length === 0 ? 'File d\'attente vide' : 'Aucun dossier trouve'}
                            </div>
                            <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginTop: '0.5rem' }}>
                                {claims.length === 0 ? 'Tous les sinistres ont ete traites automatiquement' : 'Modifiez vos filtres de recherche'}
                            </div>
                        </div>
                    )}

                    {filtered.map((claim, i) => {
                        const score = claim.analysis?.finalScore ?? 50
                        const priority = getPriority(score)
                        const pc = PRIORITY_CONFIG[priority]
                        return (
                            <div key={claim.id}
                                style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.4fr 1.8fr 0.9fr 1.1fr 0.9fr 0.9fr 90px', padding: '1rem 1.5rem', borderBottom: i < filtered.length - 1 ? `1px solid ${cardBorder}` : 'none', alignItems: 'center', transition: 'background 0.15s' }}
                                onMouseEnter={e => e.currentTarget.style.backgroundColor = rowHover}
                                onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.reference}</div>
                                <div>
                                    <div style={{ fontSize: '0.82rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{clientName(claim.client)}</div>
                                    <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.client?.company || ''}</div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.82rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.equipment?.name || '-'}</div>
                                    {claim.equipment?.type && <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claim.equipment.type}</div>}
                                </div>
                                <div style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(claim.incidentDate).toLocaleDateString('fr-FR')}</div>
                                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                    {claim.claimedAmount != null ? `${claim.claimedAmount.toLocaleString('fr-FR')} DA` : '—'}
                                </div>
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
                                    style={{ padding: '0.45rem 0.9rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                                    Traiter →
                                </button>
                            </div>
                        )
                    })}
                </div>

                <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: cardBg, borderRadius: 10, border: `1px solid ${cardBorder}`, fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    ℹ Ces dossiers sont dans la zone grise (score 30–69). Votre decision est finale et irreversible. Le client est notifie automatiquement.
                </div>
            </div>
        </div>
    )
}