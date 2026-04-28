import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

/**
 * EquipmentPage — CLIENT only
 *
 * Covers items 12–16:
 *   POST   /equipment            → register new machine
 *   GET    /equipment            → paginated list with search + type filter
 *   GET    /equipment/:id        → full details + last 5 claims
 *   PATCH  /equipment/:id        → edit name, location, manufacturer, model, dates
 *   DELETE /equipment/:id        → soft delete (isActive = false)
 *
 * Backend shape:
 *   equipment.name, .type, .manufacturer, .model, .serialNumber
 *   equipment.commissionDate, .lastMaintenanceDate, .location, .isActive
 *   equipment._count.claims
 *   equipment.claims[]   (on detail view)
 *
 * Route in App.jsx to add:
 *   <Route path="/client/equipment" element={<ProtectedRoute allowedRole="CLIENT"><EquipmentPage /></ProtectedRoute>} />
 *
 * Also add to Sidebar CLIENT_ITEMS:
 *   { key: '/client/equipment', label: 'Mes équipements', icon: '⚙' }
 */

const EQUIPMENT_TYPES = [
    'Pompe Industrielle',
    'Compresseur',
    'Moteur Electrique',
    'Generateur',
    'Turbine',
    'Pompe Hydraulique',
]

const TYPE_ICONS = {
    'Pompe Industrielle': '💧',
    'Compresseur': '🔧',
    'Moteur Electrique': '⚡',
    'Generateur': '🔋',
    'Turbine': '🌀',
    'Pompe Hydraulique': '⚙',
}

const CLAIM_STATUS_CONFIG = {
    APPROVED: { label: 'Approuvé', color: '#1A7A4A', bg: '#F0FAF4' },
    REJECTED: { label: 'Rejeté', color: '#C0392B', bg: '#FDF2F2' },
    PENDING: { label: 'En attente', color: '#7D6608', bg: '#FEF9E7' },
    ANALYZING: { label: 'Analyse', color: '#1A5276', bg: '#EBF5FB' },
    HUMAN_REVIEW: { label: 'Révision', color: '#1A5276', bg: '#EBF5FB' },
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function extractEquipment(responseData) {
    const inner = responseData?.data ?? responseData
    const arr = inner?.data ?? inner
    return Array.isArray(arr) ? arr : []
}

function today() {
    return new Date().toISOString().split('T')[0]
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function Modal({ open, title, onClose, children, dark }) {
    if (!open) return null
    const cardBg = dark ? '#111C30' : 'white'
    const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
    const textMain = dark ? 'white' : '#0F2347'
    return (
        <div style={{ position: 'fixed', inset: 0, zIndex: 500, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1rem' }}>
            <div onClick={onClose} style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)' }} />
            <div style={{ position: 'relative', backgroundColor: cardBg, border: `1px solid ${cardBorder}`, borderRadius: 16, padding: '2rem', width: '100%', maxWidth: 560, maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 24px 60px rgba(0,0,0,0.3)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>{title}</h2>
                    <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.4rem', color: dark ? '#5A7A9A' : '#9CA3AF', lineHeight: 1 }}>×</button>
                </div>
                {children}
            </div>
        </div>
    )
}

function EquipmentForm({ initial, onSubmit, loading, error, dark, submitLabel }) {
    const [form, setForm] = useState({
        name: initial?.name || '',
        type: initial?.type || '',
        manufacturer: initial?.manufacturer || '',
        model: initial?.model || '',
        serialNumber: initial?.serialNumber || '',
        commissionDate: initial?.commissionDate
            ? new Date(initial.commissionDate).toISOString().split('T')[0]
            : '',
        lastMaintenanceDate: initial?.lastMaintenanceDate
            ? new Date(initial.lastMaintenanceDate).toISOString().split('T')[0]
            : '',
        location: initial?.location || '',
    })

    const textMain = dark ? 'white' : '#0F2347'
    const textSub = dark ? '#5A7A9A' : '#9CA3AF'
    const inputBg = dark ? '#0D1626' : '#F9FAFB'
    const inputBorder = dark ? '#1E2D45' : '#E5E7EB'
    const isEdit = !!initial

    const inputStyle = {
        width: '100%', padding: '0.65rem 0.9rem',
        border: `1.5px solid ${inputBorder}`, borderRadius: 7,
        fontSize: '0.88rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
        outline: 'none', backgroundColor: inputBg, color: textMain,
        boxSizing: 'border-box',
    }
    const labelStyle = {
        display: 'block', fontSize: '0.72rem', fontWeight: 600,
        textTransform: 'uppercase', letterSpacing: '0.06em',
        color: textSub, marginBottom: '0.35rem',
        fontFamily: 'Helvetica Neue, Arial, sans-serif',
    }

    const handleSubmit = (e) => {
        e.preventDefault()
        onSubmit(form)
    }

    return (
        <form onSubmit={handleSubmit}>
            {error && (
                <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 6, padding: '0.65rem 0.9rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>
                    ⚠ {error}
                </div>
            )}

            {/* Name */}
            <div style={{ marginBottom: '0.85rem' }}>
                <label style={labelStyle}>Nom de l'équipement *</label>
                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
                    style={inputStyle} placeholder="Ex: Compresseur Atlas Copco GA-55" required
                    onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
            </div>

            {/* Type — locked on edit */}
            <div style={{ marginBottom: '0.85rem' }}>
                <label style={labelStyle}>Type d'équipement *</label>
                {isEdit ? (
                    <div style={{ ...inputStyle, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', color: textSub, cursor: 'not-allowed', display: 'flex', alignItems: 'center' }}>
                        {TYPE_ICONS[form.type] || '⚙'} {form.type}
                        <span style={{ marginLeft: '0.5rem', fontSize: '0.68rem', color: textSub }}>(non modifiable)</span>
                    </div>
                ) : (
                    <select value={form.type} onChange={e => setForm({ ...form, type: e.target.value })}
                        style={inputStyle} required
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder}>
                        <option value="">Sélectionnez un type</option>
                        {EQUIPMENT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                )}
            </div>

            {/* Manufacturer + Model */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
                <div>
                    <label style={labelStyle}>Fabricant</label>
                    <input value={form.manufacturer} onChange={e => setForm({ ...form, manufacturer: e.target.value })}
                        style={inputStyle} placeholder="Ex: Atlas Copco"
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                </div>
                <div>
                    <label style={labelStyle}>Modèle</label>
                    <input value={form.model} onChange={e => setForm({ ...form, model: e.target.value })}
                        style={inputStyle} placeholder="Ex: GA-55"
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                </div>
            </div>

            {/* Serial number — locked on edit */}
            <div style={{ marginBottom: '0.85rem' }}>
                <label style={labelStyle}>Numéro de série *</label>
                {isEdit ? (
                    <div style={{ ...inputStyle, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', color: textSub, cursor: 'not-allowed' }}>
                        {form.serialNumber} <span style={{ fontSize: '0.68rem' }}>(non modifiable)</span>
                    </div>
                ) : (
                    <input value={form.serialNumber} onChange={e => setForm({ ...form, serialNumber: e.target.value.toUpperCase() })}
                        style={inputStyle} placeholder="Ex: AC-GA55-2019-001" required
                        pattern="[A-Z0-9\-]+"
                        title="Majuscules, chiffres et tirets uniquement"
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                )}
            </div>

            {/* Commission date + Last maintenance */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
                <div>
                    <label style={labelStyle}>Date de mise en service *</label>
                    <input type="date" value={form.commissionDate}
                        max={today()}
                        onChange={e => setForm({ ...form, commissionDate: e.target.value })}
                        style={inputStyle} required
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                </div>
                <div>
                    <label style={labelStyle}>Dernière maintenance</label>
                    <input type="date" value={form.lastMaintenanceDate}
                        max={today()}
                        onChange={e => setForm({ ...form, lastMaintenanceDate: e.target.value })}
                        style={inputStyle}
                        onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                </div>
            </div>

            {/* Location */}
            <div style={{ marginBottom: '1.25rem' }}>
                <label style={labelStyle}>Emplacement</label>
                <input value={form.location} onChange={e => setForm({ ...form, location: e.target.value })}
                    style={inputStyle} placeholder="Ex: Usine Boumerdès — Bâtiment B2"
                    onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
            </div>

            <button type="submit" disabled={loading}
                style={{ width: '100%', padding: '0.8rem', background: loading ? '#9CA3AF' : 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', letterSpacing: '0.05em' }}>
                {loading ? 'Enregistrement...' : submitLabel || 'Enregistrer'}
            </button>
        </form>
    )
}

function EquipmentDetailPanel({ equipment, onClose, onEdit, onDeactivate, dark }) {
    const cardBg = dark ? '#0D1626' : '#F7F8FC'
    const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
    const textMain = dark ? 'white' : '#0F2347'
    const textSub = dark ? '#5A7A9A' : '#9CA3AF'

    const icon = TYPE_ICONS[equipment.type] || '⚙'
    const claimCount = equipment._count?.claims ?? 0

    return (
        <div>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ width: 56, height: 56, borderRadius: 12, background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.8rem', flexShrink: 0 }}>
                    {icon}
                </div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{equipment.name}</div>
                    <div style={{ fontSize: '0.78rem', color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{equipment.type}</div>
                    <div style={{ marginTop: '0.4rem' }}>
                        <span style={{ padding: '0.2rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: equipment.isActive ? '#F0FAF4' : '#FDF2F2', color: equipment.isActive ? '#1A7A4A' : '#C0392B', border: `1px solid ${equipment.isActive ? '#B8E4CA' : '#EBCECE'}` }}>
                            {equipment.isActive ? '● Actif' : '● Désactivé'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Info grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.25rem' }}>
                {[
                    ['Numéro de série', equipment.serialNumber || '—'],
                    ['Fabricant', equipment.manufacturer || '—'],
                    ['Modèle', equipment.model || '—'],
                    ['Total sinistres', `${claimCount} sinistre(s)`],
                    ['Mise en service', equipment.commissionDate ? new Date(equipment.commissionDate).toLocaleDateString('fr-FR') : '—'],
                    ['Dernière maint.', equipment.lastMaintenanceDate ? new Date(equipment.lastMaintenanceDate).toLocaleDateString('fr-FR') : '—'],
                ].map(([k, v]) => (
                    <div key={k} style={{ backgroundColor: cardBg, borderRadius: 8, padding: '0.75rem', border: `1px solid ${cardBorder}` }}>
                        <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                        <div style={{ fontSize: '0.85rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{v}</div>
                    </div>
                ))}
            </div>

            {/* Location */}
            {equipment.location && (
                <div style={{ backgroundColor: cardBg, borderRadius: 8, padding: '0.75rem', border: `1px solid ${cardBorder}`, marginBottom: '1.25rem' }}>
                    <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>Emplacement</div>
                    <div style={{ fontSize: '0.85rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>📍 {equipment.location}</div>
                </div>
            )}

            {/* Last 5 claims */}
            {equipment.claims && equipment.claims.length > 0 && (
                <div style={{ marginBottom: '1.25rem' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.6rem' }}>
                        Derniers sinistres ({equipment.claims.length})
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                        {equipment.claims.map(c => {
                            const sc = CLAIM_STATUS_CONFIG[c.status] || CLAIM_STATUS_CONFIG.PENDING
                            return (
                                <div key={c.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.55rem 0.75rem', backgroundColor: cardBg, borderRadius: 7, border: `1px solid ${cardBorder}` }}>
                                    <div>
                                        <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{c.reference}</div>
                                        <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(c.createdAt).toLocaleDateString('fr-FR')}</div>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                        <div style={{ fontSize: '0.72rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>
                                            {c.claimedAmount != null ? c.claimedAmount.toLocaleString('fr-FR') + ' DA' : '—'}
                                        </div>
                                        <span style={{ padding: '0.15rem 0.5rem', borderRadius: 20, fontSize: '0.65rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sc.bg, color: sc.color }}>
                                            {sc.label}
                                        </span>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', gap: '0.75rem', paddingTop: '1rem', borderTop: `1px solid ${cardBorder}` }}>
                <button onClick={onEdit}
                    style={{ flex: 1, padding: '0.7rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.84rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                    ✏ Modifier
                </button>
                {equipment.isActive && (
                    <button onClick={onDeactivate}
                        style={{ padding: '0.7rem 1rem', background: 'none', color: '#C0392B', border: '1.5px solid #EBCECE', borderRadius: 8, fontSize: '0.84rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                        Désactiver
                    </button>
                )}
            </div>
        </div>
    )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function EquipmentPage() {
    const navigate = useNavigate()
    const [dark, toggleDark] = useDarkMode()

    // List state
    const [equipment, setEquipment] = useState([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const [typeFilter, setTypeFilter] = useState('')
    const [page, setPage] = useState(1)
    const [pagination, setPagination] = useState(null)

    // Modal state
    const [showAdd, setShowAdd] = useState(false)
    const [showDetail, setShowDetail] = useState(null)   // equipment object
    const [showEdit, setShowEdit] = useState(null)   // equipment object
    const [showConfirmDeactivate, setShowConfirmDeactivate] = useState(null)

    // Form state
    const [formLoading, setFormLoading] = useState(false)
    const [formError, setFormError] = useState('')
    const [detailLoading, setDetailLoading] = useState(false)

    // ── Fetch list ─────────────────────────────────────────────────────────────
    const fetchEquipment = async () => {
        setLoading(true)
        try {
            const params = new URLSearchParams({ page, limit: 9 })
            if (search) params.append('search', search)
            if (typeFilter) params.append('type', typeFilter)
            const res = await api.get(`/equipment?${params}`)
            const inner = res.data?.data ?? res.data
            setEquipment(inner?.data ?? [])
            setPagination(inner?.pagination ?? null)
        } catch (err) {
            console.error('Equipment fetch error:', err)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => { fetchEquipment() }, [page, typeFilter])

    // Debounce search
    useEffect(() => {
        const t = setTimeout(() => { setPage(1); fetchEquipment() }, 350)
        return () => clearTimeout(t)
    }, [search])

    // ── Load full detail (with claims) ─────────────────────────────────────────
    const openDetail = async (eq) => {
        setDetailLoading(true)
        setShowDetail(eq) // show stub immediately
        try {
            const res = await api.get(`/equipment/${eq.id}`)
            const data = res.data?.data ?? res.data
            setShowDetail(data)
        } catch { /* keep stub */ }
        finally { setDetailLoading(false) }
    }

    // ── Register ───────────────────────────────────────────────────────────────
    const handleCreate = async (form) => {
        setFormError('')
        setFormLoading(true)
        try {
            await api.post('/equipment', form)
            setShowAdd(false)
            setPage(1)
            await fetchEquipment()
        } catch (err) {
            const msg = err.response?.data?.message
            setFormError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de l\'enregistrement')
        } finally {
            setFormLoading(false)
        }
    }

    // ── Update ─────────────────────────────────────────────────────────────────
    const handleUpdate = async (form) => {
        setFormError('')
        setFormLoading(true)
        try {
            const patch = {}
            if (form.name) patch.name = form.name
            if (form.manufacturer) patch.manufacturer = form.manufacturer
            if (form.model) patch.model = form.model
            if (form.location) patch.location = form.location
            if (form.commissionDate) patch.commissionDate = form.commissionDate
            if (form.lastMaintenanceDate) patch.lastMaintenanceDate = form.lastMaintenanceDate

            await api.patch(`/equipment/${showEdit.id}`, patch)
            setShowEdit(null)
            await fetchEquipment()
        } catch (err) {
            const msg = err.response?.data?.message
            setFormError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la mise à jour')
        } finally {
            setFormLoading(false)
        }
    }

    // ── Deactivate ─────────────────────────────────────────────────────────────
    const handleDeactivate = async (id) => {
        try {
            await api.delete(`/equipment/${id}`)
            setShowConfirmDeactivate(null)
            setShowDetail(null)
            await fetchEquipment()
        } catch (err) {
            console.error('Deactivate error:', err)
        }
    }

    // ── Colors ─────────────────────────────────────────────────────────────────
    const pageBg = dark ? '#0D1626' : '#F7F8FC'
    const cardBg = dark ? '#111C30' : 'white'
    const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
    const textMain = dark ? 'white' : '#0F2347'
    const textSub = dark ? '#5A7A9A' : '#9CA3AF'
    const rowHover = dark ? '#172338' : '#F9FAFB'

    return (
        <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
            <Sidebar role="CLIENT" dark={dark} />

            <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

                {/* ── Header ── */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
                    <div>
                        <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Parc machines</p>
                        <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Mes <strong>équipements</strong></h1>
                        <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                            {pagination?.total ?? equipment.length} machine(s) enregistrée(s)
                        </p>
                    </div>
                    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                        <NotificationBell dark={dark} />
                        <button onClick={toggleDark}
                            style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
                            {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
                        </button>
                        <button onClick={() => { setFormError(''); setShowAdd(true) }}
                            style={{ padding: '0.7rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', boxShadow: '0 4px 15px rgba(15,35,71,0.25)' }}>
                            + Nouveau équipement
                        </button>
                    </div>
                </div>

                {/* ── Filters ── */}
                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <input
                        placeholder="Rechercher par nom, fabricant, modèle, numéro série..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{ padding: '0.6rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', width: 340, backgroundColor: cardBg, color: textMain }}
                    />
                    <select value={typeFilter} onChange={e => { setTypeFilter(e.target.value); setPage(1) }}
                        style={{ padding: '0.6rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', backgroundColor: cardBg, color: textMain, cursor: 'pointer' }}>
                        <option value="">Tous les types</option>
                        {EQUIPMENT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                </div>

                {/* ── Grid ── */}
                {loading ? (
                    <div style={{ padding: '4rem', textAlign: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
                ) : equipment.length === 0 ? (
                    <div style={{ padding: '4rem', textAlign: 'center', backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}` }}>
                        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚙</div>
                        <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                            {search || typeFilter ? 'Aucun résultat' : 'Aucun équipement enregistré'}
                        </div>
                        <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                            {search || typeFilter ? 'Modifiez vos critères de recherche' : 'Ajoutez votre première machine industrielle'}
                        </div>
                        {!search && !typeFilter && (
                            <button onClick={() => { setFormError(''); setShowAdd(true) }}
                                style={{ padding: '0.65rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
                                + Ajouter un équipement
                            </button>
                        )}
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                        {equipment.map(eq => {
                            const icon = TYPE_ICONS[eq.type] || '⚙'
                            const claimCount = eq._count?.claims ?? 0
                            return (
                                <div key={eq.id}
                                    onClick={() => openDetail(eq)}
                                    style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${eq.isActive ? cardBorder : '#EBCECE'}`, padding: '1.25rem', cursor: 'pointer', transition: 'all 0.18s', opacity: eq.isActive ? 1 : 0.65, position: 'relative', overflow: 'hidden' }}
                                    onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = dark ? '0 8px 24px rgba(0,0,0,0.3)' : '0 8px 24px rgba(15,35,71,0.1)' }}
                                    onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}>

                                    {/* Inactive badge */}
                                    {!eq.isActive && (
                                        <div style={{ position: 'absolute', top: '0.75rem', right: '0.75rem', padding: '0.15rem 0.5rem', borderRadius: 20, fontSize: '0.62rem', fontWeight: 600, backgroundColor: '#FDF2F2', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                            Désactivé
                                        </div>
                                    )}

                                    {/* Icon + name */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
                                        <div style={{ width: 44, height: 44, borderRadius: 10, background: eq.isActive ? 'linear-gradient(135deg, #0F2347, #1A3A6B)' : '#E5E7EB', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.4rem', flexShrink: 0 }}>
                                            {icon}
                                        </div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{eq.name}</div>
                                            <div style={{ fontSize: '0.72rem', color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{eq.type}</div>
                                        </div>
                                    </div>

                                    {/* Details */}
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginBottom: '0.75rem' }}>
                                        {eq.serialNumber && (
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                                <span style={{ color: textSub }}>Série</span>
                                                <span style={{ color: textMain, fontWeight: 600, fontFamily: 'monospace' }}>{eq.serialNumber}</span>
                                            </div>
                                        )}
                                        {eq.manufacturer && (
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                                <span style={{ color: textSub }}>Fabricant</span>
                                                <span style={{ color: textMain }}>{eq.manufacturer}{eq.model ? ` ${eq.model}` : ''}</span>
                                            </div>
                                        )}
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                            <span style={{ color: textSub }}>Sinistres</span>
                                            <span style={{ color: claimCount > 0 ? '#E67E22' : textMain, fontWeight: claimCount > 0 ? 700 : 400 }}>{claimCount}</span>
                                        </div>
                                        {eq.location && (
                                            <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.15rem' }}>📍 {eq.location}</div>
                                        )}
                                    </div>

                                    {/* Footer */}
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.75rem', borderTop: `1px solid ${cardBorder}` }}>
                                        <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                            {eq.commissionDate ? `Depuis ${new Date(eq.commissionDate).getFullYear()}` : ''}
                                        </div>
                                        <div style={{ fontSize: '0.78rem', color: dark ? '#5A7A9A' : '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Voir détails →</div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                )}

                {/* ── Pagination ── */}
                {pagination && pagination.totalPages > 1 && (
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', marginTop: '1rem' }}>
                        {Array.from({ length: pagination.totalPages }, (_, i) => i + 1).map(p => (
                            <button key={p} onClick={() => setPage(p)}
                                style={{ width: 36, height: 36, borderRadius: 8, border: `1.5px solid ${p === page ? '#0F2347' : cardBorder}`, background: p === page ? '#0F2347' : cardBg, color: p === page ? 'white' : textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: p === page ? 700 : 400 }}>
                                {p}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Modal: Add equipment ── */}
            <Modal open={showAdd} title="Enregistrer un équipement" onClose={() => setShowAdd(false)} dark={dark}>
                <EquipmentForm
                    onSubmit={handleCreate}
                    loading={formLoading}
                    error={formError}
                    dark={dark}
                    submitLabel="Enregistrer l'équipement"
                />
            </Modal>

            {/* ── Modal: Equipment detail ── */}
            <Modal open={!!showDetail} title="Détails de l'équipement" onClose={() => setShowDetail(null)} dark={dark}>
                {showDetail && (
                    detailLoading
                        ? <div style={{ textAlign: 'center', padding: '2rem', color: dark ? '#5A7A9A' : '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
                        : (
                            <EquipmentDetailPanel
                                equipment={showDetail}
                                dark={dark}
                                onClose={() => setShowDetail(null)}
                                onEdit={() => { setShowDetail(null); setFormError(''); setShowEdit(showDetail) }}
                                onDeactivate={() => { setShowDetail(null); setShowConfirmDeactivate(showDetail) }}
                            />
                        )
                )}
            </Modal>

            {/* ── Modal: Edit equipment ── */}
            <Modal open={!!showEdit} title="Modifier l'équipement" onClose={() => setShowEdit(null)} dark={dark}>
                {showEdit && (
                    <EquipmentForm
                        initial={showEdit}
                        onSubmit={handleUpdate}
                        loading={formLoading}
                        error={formError}
                        dark={dark}
                        submitLabel="Enregistrer les modifications"
                    />
                )}
            </Modal>

            {/* ── Modal: Confirm deactivate ── */}
            <Modal open={!!showConfirmDeactivate} title="Désactiver l'équipement" onClose={() => setShowConfirmDeactivate(null)} dark={dark}>
                {showConfirmDeactivate && (
                    <div>
                        <div style={{ backgroundColor: dark ? '#2B0D0D' : '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 10, padding: '1rem', marginBottom: '1.25rem' }}>
                            <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>
                                ⚠ Désactiver {showConfirmDeactivate.name}
                            </div>
                            <div style={{ fontSize: '0.82rem', color: dark ? '#F5A0A0' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.5 }}>
                                L'équipement sera désactivé. L'historique des sinistres est conservé. Aucun nouveau sinistre ne pourra être soumis pour cette machine.
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: '0.75rem' }}>
                            <button onClick={() => setShowConfirmDeactivate(null)}
                                style={{ flex: 1, padding: '0.7rem', background: 'none', border: `1.5px solid ${dark ? '#1E2D45' : '#E5E7EB'}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', color: dark ? '#5A7A9A' : '#9CA3AF' }}>
                                Annuler
                            </button>
                            <button onClick={() => handleDeactivate(showConfirmDeactivate.id)}
                                style={{ flex: 1, padding: '0.7rem', background: 'linear-gradient(135deg, #C0392B, #E74C3C)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                                Confirmer la désactivation
                            </button>
                        </div>
                    </div>
                )}
            </Modal>
        </div>
    )
}