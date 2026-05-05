import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

/* ─────────────────────────────────────────────────────────────
   DATA ANATOMY CANVAS
───────────────────────────────────────────────────────────── */
function AnatomyCanvas({ assembled }) {
    const mountRef = useRef(null)
    const stateRef = useRef({ assembled: false })

    useEffect(() => { stateRef.current.assembled = assembled }, [assembled])

    useEffect(() => {
        let animId, renderer, THREE
        const parts = []
        let badgeMesh = null
        let badgeScale = 0

        const init = async () => {
            THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js').catch(() => null)
            if (!THREE || !mountRef.current) return

            const W = mountRef.current.clientWidth
            const H = mountRef.current.clientHeight

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
            renderer.setSize(W, H)
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
            renderer.setClearColor(0x000000, 0)
            mountRef.current.appendChild(renderer.domElement)

            const scene = new THREE.Scene()
            const camera = new THREE.PerspectiveCamera(46, W / H, 0.1, 200)
            camera.position.set(0, 0, 36)

            scene.add(new THREE.AmbientLight(0xffffff, 0.5))
            const goldL = new THREE.DirectionalLight(0xE8C97A, 2.8)
            goldL.position.set(-5, 6, 10); scene.add(goldL)
            const blueL = new THREE.DirectionalLight(0x4488CC, 1.2)
            blueL.position.set(8, -4, -6); scene.add(blueL)

            const metalMat = new THREE.MeshStandardMaterial({ color: 0x1A2A45, roughness: 0.55, metalness: 0.88 })
            const darkMat = new THREE.MeshStandardMaterial({ color: 0x0F1A2E, roughness: 0.65, metalness: 0.92 })
            const goldMat = new THREE.MeshStandardMaterial({ color: 0xC9A84C, roughness: 0.28, metalness: 0.96 })

            const pumpDefs = [
                { geo: new THREE.TorusGeometry(0.72, 0.38, 16, 48), mat: metalMat, aPos: [0, 0, 0], aRot: [Math.PI / 2, 0, 0], dir: [0, 0, 1.8] },
                { geo: new THREE.CylinderGeometry(1.02, 1.02, 0.28, 40), mat: metalMat, aPos: [0, 0, 0], aRot: [0, 0, 0], dir: [0, 2.5, 0] },
                { geo: new THREE.CylinderGeometry(0.22, 0.22, 0.85, 16), mat: darkMat, aPos: [0, 0.9, 0], aRot: [0, 0, 0], dir: [0, 4.5, 1.5] },
                { geo: new THREE.CylinderGeometry(0.32, 0.32, 0.08, 16), mat: goldMat, aPos: [0, 1.28, 0], aRot: [0, 0, 0], dir: [1.5, 5.5, 0] },
                { geo: new THREE.CylinderGeometry(0.26, 0.26, 0.9, 16), mat: darkMat, aPos: [-1.0, 0, 0], aRot: [0, 0, Math.PI / 2], dir: [-5, 0.5, 0] },
                { geo: new THREE.CylinderGeometry(0.36, 0.36, 0.08, 16), mat: goldMat, aPos: [-1.42, 0, 0], aRot: [0, 0, Math.PI / 2], dir: [-7, 1, 0] },
                { geo: new THREE.CylinderGeometry(0.18, 0.18, 0.28, 12), mat: darkMat, aPos: [1.18, 0, 0], aRot: [0, 0, Math.PI / 2], dir: [4, -1, 0] },
                { geo: new THREE.CylinderGeometry(0.30, 0.30, 0.52, 20), mat: metalMat, aPos: [1.65, 0, 0], aRot: [0, 0, Math.PI / 2], dir: [5.5, 0, 1] },
                { geo: new THREE.BoxGeometry(0.95, 0.72, 0.72), mat: metalMat, aPos: [2.35, 0, 0], aRot: [0, 0, 0], dir: [7, -0.5, -1] },
                { geo: new THREE.BoxGeometry(3.6, 0.12, 0.88), mat: darkMat, aPos: [0.6, -0.52, 0], aRot: [0, 0, 0], dir: [0, -4, 0] },
                { geo: new THREE.BoxGeometry(0.28, 0.18, 0.88), mat: goldMat, aPos: [-1.5, -0.65, 0], aRot: [0, 0, 0], dir: [-3, -5, 0] },
                { geo: new THREE.BoxGeometry(0.28, 0.18, 0.88), mat: goldMat, aPos: [1.8, -0.65, 0], aRot: [0, 0, 0], dir: [3.5, -5, 0] },
            ]

            const SCALE = 0.84
            pumpDefs.forEach((def, i) => {
                const mesh = new THREE.Mesh(def.geo, def.mat)
                const aPos = new THREE.Vector3(...def.aPos).multiplyScalar(SCALE)
                mesh.rotation.set(...def.aRot)
                const explodedPos = aPos.clone().add(new THREE.Vector3(...def.dir).multiplyScalar(1.6))
                mesh.position.copy(explodedPos)
                scene.add(mesh)
                parts.push({ mesh, aPos, explodedPos, phase: i * 0.18 })
            })

            const badgeMat2 = new THREE.MeshStandardMaterial({ color: 0xC0392B, roughness: 0.3, metalness: 0.7, emissive: 0xC0392B, emissiveIntensity: 0.4 })
            badgeMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 0.9, 0.18, 32), badgeMat2)
            badgeMesh.position.set(0, 0, 1.2); badgeMesh.scale.setScalar(0); scene.add(badgeMesh)

            const connectors = []
            parts.forEach(p => {
                const geo = new THREE.BufferGeometry().setFromPoints([p.explodedPos.clone(), p.aPos.clone()])
                const mat = new THREE.LineBasicMaterial({ color: 0xC9A84C, transparent: true, opacity: 0 })
                scene.add(new THREE.Line(geo, mat))
                connectors.push({ mat })
            })

            const pCount = 120, pPos = new Float32Array(pCount * 3)
            for (let i = 0; i < pCount; i++) { pPos[i * 3] = (Math.random() - .5) * 50; pPos[i * 3 + 1] = (Math.random() - .5) * 32; pPos[i * 3 + 2] = (Math.random() - .5) * 20 }
            const pGeo = new THREE.BufferGeometry()
            pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3))
            scene.add(new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0xC9A84C, size: 0.07, transparent: true, opacity: 0.25 })))

            const grid = new THREE.GridHelper(30, 30, 0xC9A84C, 0xC9A84C)
            grid.material.transparent = true; grid.material.opacity = 0.03; grid.position.y = -4.5; scene.add(grid)

            const handleResize = () => {
                if (!mountRef.current) return
                const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight
                camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h)
            }
            window.addEventListener('resize', handleResize)

            let t = 0
            const animate = () => {
                animId = requestAnimationFrame(animate); t += 0.01
                const isA = stateRef.current.assembled
                parts.forEach(p => {
                    p.mesh.position.lerp(isA ? p.aPos : p.explodedPos, 0.045)
                    if (!isA) { p.mesh.position.y += Math.sin(t * 0.8 + p.phase) * 0.006; p.mesh.rotation.y += 0.003 }
                })
                const cop = isA ? Math.min((connectors[0]?.mat.opacity ?? 0) + 0.015, 0.3) : Math.max((connectors[0]?.mat.opacity ?? 0) - 0.01, 0)
                connectors.forEach(c => { c.mat.opacity = cop })
                if (isA) { badgeScale = Math.min(badgeScale + 0.04, 1); badgeMesh.scale.setScalar(badgeScale * (1 + 0.05 * Math.sin(t * 2))) }
                else { badgeScale = Math.max(badgeScale - 0.04, 0); badgeMesh.scale.setScalar(badgeScale) }
                scene.rotation.y = Math.sin(t * 0.06) * 0.12
                renderer.render(scene, camera)
            }
            animate()
        }

        init()
        return () => {
            if (animId) cancelAnimationFrame(animId)
            if (renderer) { renderer.dispose(); if (mountRef.current && renderer.domElement.parentNode === mountRef.current) mountRef.current.removeChild(renderer.domElement) }
        }
    }, [])

    return <div ref={mountRef} style={{ position: 'absolute', inset: 0, zIndex: 1 }} />
}

/* ─────────────────────────────────────────────────────────────
   ANIMATED COUNTER
───────────────────────────────────────────────────────────── */
function Counter({ target, suffix = '', duration = 1800 }) {
    const [val, setVal] = useState(0)
    const ref = useRef(null)
    const started = useRef(false)
    useEffect(() => {
        const obs = new IntersectionObserver(([e]) => {
            if (e.isIntersecting && !started.current) {
                started.current = true
                const start = performance.now()
                const tick = now => { const p = Math.min((now - start) / duration, 1); setVal(Math.round((1 - Math.pow(1 - p, 3)) * target)); if (p < 1) requestAnimationFrame(tick) }
                requestAnimationFrame(tick)
            }
        }, { threshold: 0.4 })
        if (ref.current) obs.observe(ref.current)
        return () => obs.disconnect()
    }, [target, duration])
    return <span ref={ref}>{val}{suffix}</span>
}

/* ─────────────────────────────────────────────────────────────
   SCORE DIAL
───────────────────────────────────────────────────────────── */
function ScoreDial({ visible, dark, textSub, textB, border }) {
    const score = 89
    const circ = 2 * Math.PI * 96
    const offset = circ - (score / 100) * circ
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ position: 'relative', width: 220, height: 220, marginBottom: '2rem' }}>
                <svg viewBox="0 0 220 220" width="220" height="220" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="110" cy="110" r="96" fill="none" stroke={dark ? '#1E2D45' : '#E5E7EB'} strokeWidth="10" />
                    <circle cx="110" cy="110" r="96" fill="none" stroke="url(#dg)" strokeWidth="10" strokeLinecap="round"
                        strokeDasharray={circ}
                        style={{ strokeDashoffset: visible ? offset : circ, transition: 'stroke-dashoffset 1.8s cubic-bezier(.22,.61,.36,1)' }} />
                    <defs>
                        <linearGradient id="dg" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#C0392B" /><stop offset="100%" stopColor="#E67E22" />
                        </linearGradient>
                    </defs>
                </svg>
                <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    <div style={{ fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '3.8rem', fontWeight: 800, background: 'linear-gradient(135deg,#C0392B,#E67E22)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', lineHeight: 1 }}>{score}</div>
                    <div style={{ fontSize: '0.7rem', color: textSub, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/ 100</div>
                </div>
            </div>
            <div style={{ padding: '0.55rem 1.5rem', borderRadius: 8, background: dark ? 'rgba(192,57,43,0.12)' : 'rgba(192,57,43,0.07)', border: '1px solid rgba(192,57,43,0.35)', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 800, fontSize: '0.82rem', letterSpacing: '0.06em', marginBottom: '1.75rem' }}>
                REJETÉ AUTOMATIQUEMENT — Score ≥ 70
            </div>
            <div style={{ width: '100%', maxWidth: 310 }}>
                {['Aucun précurseur capteurs sur 6 mois', 'Classe FALSIFIÉ — confiance 91 %', 'EXIF daté 8 mois avant l\'incident', 'Contradictions majeures dans le rapport'].map(ind => (
                    <div key={ind} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', fontSize: '0.8rem', color: textB, padding: '0.55rem 0', borderBottom: `1px solid ${border}`, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                        <span style={{ color: '#C0392B', fontWeight: 700, fontSize: '0.85rem', flexShrink: 0 }}>–</span>
                        {ind}
                    </div>
                ))}
            </div>
        </div>
    )
}

/* ─────────────────────────────────────────────────────────────
   MAIN PAGE
───────────────────────────────────────────────────────────── */
export default function LandingPageV2() {
    const navigate = useNavigate()
    const [dark, setDark] = useState(false)
    const [scrolled, setScrolled] = useState(false)
    const [assembled, setAssembled] = useState(false)
    const [visibleSections, setVisibleSections] = useState(new Set())

    useEffect(() => {
        const mq = window.matchMedia('(prefers-color-scheme: dark)')
        const saved = localStorage.getItem('darkMode')
        setDark(saved !== null ? saved === 'true' : mq.matches)
    }, [])

    const toggleDark = () => setDark(d => { localStorage.setItem('darkMode', String(!d)); return !d })

    useEffect(() => {
        const onScroll = () => { setScrolled(window.scrollY > 40); setAssembled(window.scrollY > 60) }
        window.addEventListener('scroll', onScroll)
        return () => window.removeEventListener('scroll', onScroll)
    }, [])

    useEffect(() => {
        const els = document.querySelectorAll('[data-reveal]')
        const obs = new IntersectionObserver(entries => {
            entries.forEach(e => { if (e.isIntersecting) setVisibleSections(p => new Set([...p, e.target.dataset.reveal])) })
        }, { threshold: 0.12 })
        els.forEach(el => obs.observe(el))
        return () => obs.disconnect()
    }, [])

    /* ── Theme ── */
    const bg = dark ? '#060D1A' : '#F7F8FC'
    const bgCard = dark ? '#0D1626' : '#FFFFFF'
    const bgCard2 = dark ? '#111C30' : '#F0F3FA'
    const border = dark ? 'rgba(201,168,76,0.12)' : 'rgba(15,35,71,0.08)'
    const borderHv = dark ? 'rgba(201,168,76,0.35)' : 'rgba(15,35,71,0.2)'
    const navy = '#0F2347'
    const gold = '#C9A84C'
    const textH = dark ? '#FFFFFF' : navy
    const textB = dark ? '#94A3B8' : '#4B5563'
    const textSub = dark ? '#4A6A8A' : '#9CA3AF'
    const navBg = scrolled ? (dark ? 'rgba(6,13,26,0.92)' : 'rgba(247,248,252,0.92)') : 'transparent'

    const reveal = id => ({ opacity: visibleSections.has(id) ? 1 : 0, transform: visibleSections.has(id) ? 'translateY(0)' : 'translateY(32px)', transition: 'opacity 0.75s ease, transform 0.75s ease' })
    const revealL = id => ({ opacity: visibleSections.has(id) ? 1 : 0, transform: visibleSections.has(id) ? 'translateX(0)' : 'translateX(-30px)', transition: 'opacity 0.75s ease, transform 0.75s ease' })
    const revealR = id => ({ opacity: visibleSections.has(id) ? 1 : 0, transform: visibleSections.has(id) ? 'translateX(0)' : 'translateX(30px)', transition: 'opacity 0.75s ease, transform 0.75s ease' })

    const chips = [
        { label: 'Vibration', val: '4.2 mm/s', color: '#C9A84C', top: '28%', left: '5%', delay: '0s', dur: '7s' },
        { label: 'Temp.', val: '87 °C', color: '#C0392B', top: '22%', right: '5%', delay: '1.2s', dur: '6s' },
        { label: 'Pression', val: '6.2 bar', color: '#1A7A4A', top: '62%', left: '4%', delay: '2.4s', dur: '8s' },
        { label: 'Débit', val: '48 m³/h', color: '#4A9EFF', top: '66%', right: '4%', delay: '3s', dur: '7.5s' },
    ]

    const models = [
        { num: '01', coef: '0.35', title: 'LSTM + Isolation Forest', sub: 'Anomalie capteurs', weight: '35%', color: '#4A9EFF', desc: 'Détecte les patterns anormaux dans les séries temporelles capteurs sur 6 mois. Identifie les précurseurs de pannes non documentées.' },
        { num: '02', coef: '0.25', title: 'XGBoost Classifier', sub: 'Classification panne', weight: '25%', color: '#C9A84C', desc: 'Classifie le type de défaillance déclaré et le compare aux profils historiques pour détecter les incohérences.' },
        { num: '03', coef: '0.20', title: 'BERT Multilingue', sub: 'Analyse NLP', weight: '20%', color: '#1ABC9C', desc: 'Analyse sémantique des rapports techniques. Détecte les contradictions entre la description narrative et les données réelles.' },
        { num: '04', coef: '0.20', title: 'YOLOv8 + ELA', sub: 'Vision & Forensique', weight: '20%', color: '#9B59B6', desc: 'Détecte les manipulations photographiques via Error Level Analysis et localise les dommages sur les photos soumises.' },
    ]

    const wfSteps = [
        { title: 'Réception et confirmation', desc: 'Courriel de confirmation envoyé automatiquement avec la référence unique du dossier.', time: 'T + 0 sec' },
        { title: 'Analyse IA parallèle', desc: '4 microservices FastAPI s\'exécutent simultanément via RabbitMQ. Chaque modèle retourne son score individuel.', time: 'T + 1 à 5 min' },
        { title: 'Score final et décision', desc: 'Agrégation pondérée. Score < 30 → approuvé automatiquement. Score > 70 → rejeté. Entre 30 et 69 → enquêteur assigné.', time: 'T + 5 min' },
        { title: 'Génération du rapport PDF', desc: 'Document officiel avec tous les indicateurs de fraude, instructions de recours et délai légal de 30 jours.', time: 'T + 5 min 10 sec' },
        { title: 'Notification et archivage', desc: 'Rapport transmis au client. Le département fraude est alerté en cas de rejet. Le dossier est archivé dans MinIO.', time: 'T + 5 min 23 sec' },
    ]

    const features = [
        { num: 'A', title: 'Authentification sécurisée', desc: 'JWT + rotation de refresh tokens, bcrypt, verrouillage de compte, invalidation multi-appareils.', color: '#4A9EFF' },
        { num: 'B', title: 'Tableau de bord temps réel', desc: 'Suivi en direct des sinistres, scores IA mis à jour automatiquement, notifications instantanées.', color: '#C9A84C' },
        { num: 'C', title: 'Gestion des équipements', desc: 'Registre des machines industrielles avec historique des sinistres et suivi de maintenance.', color: '#1ABC9C' },
        { num: 'D', title: 'Lettre de décision PDF', desc: 'Génération automatique de la lettre officielle de décision après chaque traitement de dossier.', color: '#9B59B6' },
        { num: 'E', title: 'Notifications temps réel', desc: 'Le client est averti à chaque changement de statut via le système de notifications.', color: '#E67E22' },
        { num: 'F', title: 'Espace enquêteur', desc: 'Interface dédiée pour réviser les dossiers en zone grise avec accès complet aux données.', color: '#C0392B' },
    ]

    return (
        <div style={{ backgroundColor: bg, minHeight: '100vh', fontFamily: 'Georgia, serif', color: textH, transition: 'background 0.4s', overflowX: 'hidden' }}>

            <style>{`
        @keyframes heroFadeUp { from{opacity:0;transform:translateY(32px)} to{opacity:1;transform:translateY(0)} }
        @keyframes chipFloat  { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
        @keyframes pulseDot   { 0%,100%{opacity:0.4;transform:scale(1)} 50%{opacity:1;transform:scale(1.35)} }
        @keyframes scanDown   { 0%{top:-60px} 100%{top:110%} }
        @keyframes gradShift  { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
        @keyframes badgePop   { 0%{opacity:0;transform:scale(0.4)} 70%{transform:scale(1.12)} 100%{opacity:1;transform:scale(1)} }
        .sbar { transition: width 1.4s cubic-bezier(.22,.61,.36,1); }
      `}</style>

            {/* ══════════ NAVBAR ══════════ */}
            <nav style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 2.5rem', height: 64, background: navBg, backdropFilter: scrolled ? 'blur(18px)' : 'none', borderBottom: scrolled ? `1px solid ${border}` : 'none', transition: 'all 0.35s' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                    <div style={{ width: 34, height: 34, borderRadius: 8, background: 'linear-gradient(135deg,#C9A84C,#E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, color: navy, fontSize: '1rem', boxShadow: '0 0 16px rgba(201,168,76,0.3)', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>F</div>
                    <div>
                        <div style={{ fontWeight: 700, fontSize: '0.9rem', color: textH, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>FraudGuard AI</div>
                        <div style={{ fontSize: '0.52rem', color: gold, letterSpacing: '0.18em', textTransform: 'uppercase', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Assurance Industrielle</div>
                    </div>
                </div>
                <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                    {['Fonctionnalités', 'Modèles IA', 'Processus'].map(l => (
                        <span key={l} style={{ fontSize: '0.82rem', color: textSub, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', transition: 'color 0.2s' }}
                            onMouseEnter={e => e.target.style.color = gold} onMouseLeave={e => e.target.style.color = textSub}>{l}</span>
                    ))}
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    <button onClick={toggleDark} style={{ background: 'none', border: `1px solid ${border}`, borderRadius: 7, padding: '0.4rem 0.85rem', cursor: 'pointer', color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', transition: 'border-color 0.2s' }}
                        onMouseEnter={e => e.currentTarget.style.borderColor = gold} onMouseLeave={e => e.currentTarget.style.borderColor = border}>
                        {dark ? 'Mode clair' : 'Mode sombre'}
                    </button>
                    <button onClick={() => navigate('/login')} style={{ padding: '0.45rem 1.1rem', borderRadius: 7, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', border: `1.5px solid ${gold}`, background: 'transparent', color: gold, transition: 'all 0.2s' }}
                        onMouseEnter={e => { e.currentTarget.style.background = gold; e.currentTarget.style.color = navy }}
                        onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = gold }}>
                        Connexion
                    </button>
                    <button onClick={() => navigate('/login')} style={{ padding: '0.45rem 1.1rem', borderRadius: 7, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', border: 'none', background: `linear-gradient(135deg,${navy},#1A3A6B)`, color: 'white', boxShadow: '0 4px 14px rgba(15,35,71,0.35)', transition: 'transform 0.15s, box-shadow 0.15s' }}
                        onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(15,35,71,0.45)' }}
                        onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 14px rgba(15,35,71,0.35)' }}>
                        S'inscrire →
                    </button>
                </div>
            </nav>

            {/* ══════════ HERO ══════════ */}
            <section style={{ position: 'relative', minHeight: '100vh', display: 'flex', alignItems: 'center', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', inset: 0, zIndex: 0, background: dark ? 'radial-gradient(ellipse 90% 80% at 60% 50%,rgba(15,35,71,0.85) 0%,rgba(6,13,26,1) 100%)' : 'radial-gradient(ellipse 90% 80% at 60% 50%,rgba(15,35,71,0.05) 0%,rgba(247,248,252,1) 100%)' }} />
                <div style={{ position: 'absolute', inset: 0, zIndex: 0, opacity: dark ? 0.03 : 0.025, backgroundImage: `linear-gradient(${dark ? 'rgba(201,168,76,1)' : 'rgba(15,35,71,1)'} 1px,transparent 1px),linear-gradient(90deg,${dark ? 'rgba(201,168,76,1)' : 'rgba(15,35,71,1)'} 1px,transparent 1px)`, backgroundSize: '52px 52px' }} />
                <div style={{ position: 'absolute', left: 0, right: 0, height: 80, background: 'linear-gradient(transparent,rgba(201,168,76,0.04),transparent)', animation: 'scanDown 9s linear infinite', zIndex: 2, pointerEvents: 'none' }} />

                {[{ top: 80, left: 24, borderTop: '1.5px solid rgba(201,168,76,0.5)', borderLeft: '1.5px solid rgba(201,168,76,0.5)' }, { top: 80, right: 24, borderTop: '1.5px solid rgba(201,168,76,0.5)', borderRight: '1.5px solid rgba(201,168,76,0.5)' }, { bottom: 24, left: 24, borderBottom: '1.5px solid rgba(201,168,76,0.5)', borderLeft: '1.5px solid rgba(201,168,76,0.5)' }, { bottom: 24, right: 24, borderBottom: '1.5px solid rgba(201,168,76,0.5)', borderRight: '1.5px solid rgba(201,168,76,0.5)' }].map((s, i) => (
                    <div key={i} style={{ position: 'absolute', width: 24, height: 24, zIndex: 5, pointerEvents: 'none', ...s }} />
                ))}

                <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '55%', zIndex: 1 }}>
                    <AnatomyCanvas assembled={assembled} />
                </div>

                {chips.map((c, i) => (
                    <div key={i} style={{ position: 'absolute', zIndex: 6, top: c.top, left: c.left, right: c.right, background: dark ? 'rgba(6,13,26,0.88)' : 'rgba(255,255,255,0.92)', border: `1px solid rgba(201,168,76,0.2)`, borderLeft: `2px solid ${c.color}`, borderRadius: 8, padding: '0.38rem 0.72rem', backdropFilter: 'blur(12px)', boxShadow: '0 4px 20px rgba(0,0,0,0.2)', animation: `chipFloat ${c.dur} ease-in-out infinite`, animationDelay: c.delay, opacity: assembled ? 0 : 1, transition: 'opacity 0.5s ease', pointerEvents: 'none' }}>
                        <div style={{ color: 'rgba(100,120,150,0.8)', fontSize: '0.55rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.1rem' }}>{c.label}</div>
                        <div style={{ color: c.color, fontWeight: 700, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{c.val}</div>
                    </div>
                ))}

                {assembled && (
                    <div style={{ position: 'absolute', right: '26%', top: '38%', zIndex: 10, background: dark ? 'rgba(6,13,26,0.92)' : 'rgba(255,255,255,0.95)', border: '2px solid #C0392B', borderRadius: 12, padding: '0.6rem 1rem', backdropFilter: 'blur(16px)', boxShadow: '0 0 32px rgba(192,57,43,0.35)', animation: 'badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) both', pointerEvents: 'none', textAlign: 'center' }}>
                        <div style={{ fontSize: '0.58rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 700, marginBottom: '0.15rem' }}>Score Fraude</div>
                        <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>67</div>
                        <div style={{ fontSize: '0.6rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100 — Zone grise</div>
                    </div>
                )}

                <div style={{ position: 'relative', zIndex: 10, width: '45%', padding: '0 0 0 5rem', paddingTop: '4rem' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.35rem 1rem', borderRadius: 20, border: `1px solid rgba(201,168,76,0.35)`, background: dark ? 'rgba(201,168,76,0.08)' : 'rgba(201,168,76,0.1)', marginBottom: '1.75rem', animation: 'heroFadeUp 0.7s ease 0.1s both' }}>
                        <span style={{ width: 7, height: 7, borderRadius: '50%', backgroundColor: gold, display: 'inline-block', animation: 'pulseDot 2s ease-in-out infinite' }} />
                        <span style={{ fontSize: '0.7rem', color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 600 }}>Détection multimodale — Algérie</span>
                    </div>

                    <h1 style={{ fontSize: 'clamp(2rem,4vw,3.5rem)', fontWeight: 400, lineHeight: 1.15, letterSpacing: '-0.025em', marginBottom: '1.25rem', animation: 'heroFadeUp 0.7s ease 0.25s both', color: textH }}>
                        Disséquez chaque<br />sinistre.{' '}
                        <em style={{ fontStyle: 'normal', fontWeight: 700, background: 'linear-gradient(135deg,#C9A84C 0%,#E8C97A 50%,#C9A84C 100%)', backgroundSize: '200% 200%', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', animation: 'gradShift 4s ease infinite' }}>Révélez la vérité.</em>
                    </h1>

                    <p style={{ fontSize: '0.95rem', color: textB, lineHeight: 1.8, marginBottom: '2rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', animation: 'heroFadeUp 0.7s ease 0.4s both', maxWidth: 420 }}>
                        FraudGuard AI décompose chaque machine, lit ses données capteurs, son rapport et ses photos — puis reconstitue un score de fraude précis en quelques secondes.
                    </p>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem', animation: 'heroFadeUp 0.7s ease 0.5s both', opacity: assembled ? 0 : 1, transition: 'opacity 0.4s' }}>
                        <div style={{ width: 1, height: 28, background: `linear-gradient(to bottom,${gold},transparent)` }} />
                        <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.1em', textTransform: 'uppercase' }}>Défilez pour assembler</span>
                    </div>

                    <div style={{ opacity: assembled ? 1 : 0, transform: assembled ? 'translateY(0)' : 'translateY(12px)', transition: 'all 0.5s ease 0.2s', marginBottom: '2rem' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.6rem', padding: '0.5rem 1rem', borderRadius: 8, background: dark ? 'rgba(192,57,43,0.12)' : 'rgba(192,57,43,0.07)', border: '1px solid rgba(192,57,43,0.3)' }}>
                            <span style={{ fontSize: '0.78rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>Anomalies détectées — Révision humaine requise</span>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', animation: 'heroFadeUp 0.7s ease 0.55s both' }}>
                        <button onClick={() => navigate('/login')} style={{ padding: '0.82rem 1.8rem', borderRadius: 10, background: `linear-gradient(135deg,${navy},#1A3A6B)`, color: 'white', border: 'none', fontSize: '0.88rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', boxShadow: '0 6px 24px rgba(15,35,71,0.38)', letterSpacing: '0.04em', transition: 'transform 0.15s, box-shadow 0.15s' }}
                            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 12px 30px rgba(15,35,71,0.5)' }}
                            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 6px 24px rgba(15,35,71,0.38)' }}>
                            Accéder à la plateforme →
                        </button>
                        <button onClick={() => navigate('/login')} style={{ padding: '0.82rem 1.8rem', borderRadius: 10, background: 'transparent', color: gold, border: `1.5px solid ${gold}`, fontSize: '0.88rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', letterSpacing: '0.04em', transition: 'all 0.2s' }}
                            onMouseEnter={e => { e.currentTarget.style.background = gold; e.currentTarget.style.color = navy }}
                            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = gold }}>
                            Créer un compte
                        </button>
                    </div>
                </div>

                <div style={{ position: 'absolute', bottom: 32, left: '22.5%', transform: 'translateX(-50%)', zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem', opacity: 0.35 }}>
                    <span style={{ fontSize: '0.6rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.14em', textTransform: 'uppercase' }}>Défiler</span>
                    <div style={{ width: 1, height: 28, background: `linear-gradient(to bottom,${gold},transparent)` }} />
                </div>
            </section>

            {/* ══════════ STATS STRIP ══════════ */}
            <section data-reveal="stats" style={{ ...reveal('stats'), backgroundColor: dark ? '#0A1628' : navy, padding: '2.5rem 0', borderTop: `1px solid ${dark ? 'rgba(201,168,76,0.15)' : 'rgba(255,255,255,0.1)'}`, borderBottom: `1px solid ${dark ? 'rgba(201,168,76,0.15)' : 'rgba(255,255,255,0.1)'}` }}>
                <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '2rem', padding: '0 2rem' }}>
                    {[{ value: 4, suffix: ' modèles', label: 'Modèles IA en parallèle' }, { value: 3, suffix: ' types', label: 'Formats de données analysés' }, { value: 100, suffix: '', label: 'Score de fraude sur 100 points' }, { value: 30, suffix: 's', label: 'Temps d\'analyse estimé' }].map(s => (
                        <div key={s.label} style={{ textAlign: 'center' }}>
                            <div style={{ fontSize: '2.4rem', fontWeight: 700, color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}><Counter target={s.value} suffix={s.suffix} /></div>
                            <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.45)', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.4rem', letterSpacing: '0.04em' }}>{s.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ SCORE FORMULA + DIAL ══════════ */}
            <section style={{ padding: '8rem clamp(1.5rem,5vw,4.5rem)', background: dark ? `linear-gradient(180deg,${bg} 0%,#0A1628 100%)` : 'linear-gradient(180deg,#F7F8FC 0%,#EEF2FF 100%)' }}>
                <div style={{ maxWidth: 1100, margin: '0 auto' }}>
                    <div data-reveal="score-head" style={{ ...revealL('score-head'), marginBottom: '1rem', fontSize: '0.73rem', color: gold, letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score de fraude</div>
                    <h2 data-reveal="score-h2" style={{ ...revealL('score-h2'), fontSize: 'clamp(2rem,3.5vw,3rem)', fontWeight: 700, letterSpacing: '-0.025em', lineHeight: 1.1, maxWidth: 540, marginBottom: '1.5rem', color: textH, transitionDelay: '0.1s' }}>
                        Un score. Une décision.<br />Tout est explicable.
                    </h2>
                    <p data-reveal="score-p" style={{ ...revealL('score-p'), color: textB, fontSize: '0.95rem', lineHeight: 1.75, maxWidth: 420, marginBottom: '4.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', transitionDelay: '0.2s' }}>
                        Chaque indicateur est traçable et auditable — essentiel pour la défense juridique des décisions de rejet.
                    </p>

                    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '3.5rem', alignItems: 'start' }}>

                        {/* Formula card */}
                        <div data-reveal="score-formula" style={{ ...revealL('score-formula'), backgroundColor: bgCard, border: `1px solid ${border}`, borderRadius: 20, padding: '2.5rem', position: 'relative', overflow: 'hidden' }}>
                            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg,#4A9EFF,#C9A84C,#1ABC9C,#9B59B6)` }} />
                            <div style={{ fontSize: '0.72rem', color: textSub, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '1.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                Formule d'agrégation pondérée
                            </div>
                            {models.map(({ coef, sub, color, weight }, i) => (
                                <div key={sub} style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.1rem' }}>
                                    <span style={{ minWidth: 44, fontSize: '1.1rem', fontWeight: 800, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{coef}</span>
                                    <div style={{ flex: 1, height: 5, background: dark ? '#1E2D45' : '#E5E7EB', borderRadius: 3, overflow: 'hidden' }}>
                                        <div className="sbar" style={{ height: '100%', borderRadius: 3, background: `linear-gradient(90deg,${color},${color}88)`, width: visibleSections.has('score-formula') ? `${parseInt(weight) * 2.5}%` : '0%', transitionDelay: `${i * 0.12}s` }} />
                                    </div>
                                    <span style={{ fontSize: '0.82rem', color: textB, minWidth: 160, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{sub}</span>
                                </div>
                            ))}
                            <div style={{ marginTop: '2rem', paddingTop: '1.75rem', borderTop: `1px solid ${border}` }}>
                                <div style={{ fontSize: '0.72rem', color: textSub, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Seuils de décision automatique</div>
                                <div style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap' }}>
                                    {[{ range: '0 – 29', label: 'Approuvé', color: '#1A7A4A', bg: dark ? 'rgba(26,122,74,0.12)' : 'rgba(26,122,74,0.08)' }, { range: '30 – 69', label: 'Révision humaine', color: '#F39C12', bg: dark ? 'rgba(243,156,18,0.12)' : 'rgba(243,156,18,0.08)' }, { range: '70 – 100', label: 'Rejeté', color: '#C0392B', bg: dark ? 'rgba(192,57,43,0.12)' : 'rgba(192,57,43,0.08)' }].map(t => (
                                        <div key={t.range} style={{ padding: '0.35rem 0.85rem', borderRadius: 7, background: t.bg, border: `1px solid ${t.color}44`, fontSize: '0.75rem', color: t.color, fontWeight: 700, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                            {t.range} → {t.label}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Score dial */}
                        <div data-reveal="score-dial" style={{ ...revealR('score-dial') }}>
                            <ScoreDial visible={visibleSections.has('score-dial')} dark={dark} textSub={textSub} textB={textB} border={border} />
                        </div>
                    </div>
                </div>
            </section>

            {/* ══════════ AI MODELS ══════════ */}
            <section data-reveal="models" style={{ ...reveal('models'), padding: '7rem 2rem', maxWidth: 1100, margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <div style={{ fontSize: '0.7rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>Intelligence artificielle</div>
                    <h2 style={{ fontSize: 'clamp(1.8rem,4vw,2.8rem)', fontWeight: 400, color: textH, letterSpacing: '-0.02em' }}>Quatre modèles. <strong>Une décision.</strong></h2>
                    <p style={{ color: textB, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.95rem', maxWidth: 520, margin: '0.75rem auto 0' }}>Chaque modèle analyse une dimension différente du sinistre. Le score final est une fusion pondérée des quatre évaluations.</p>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: '1.25rem' }}>
                    {models.map(m => (
                        <div key={m.title} style={{ backgroundColor: bgCard, borderRadius: 16, border: `1px solid ${border}`, padding: '1.75rem', transition: 'border-color 0.25s,transform 0.25s,box-shadow 0.25s', position: 'relative', overflow: 'hidden' }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = m.color; e.currentTarget.style.transform = 'translateY(-5px)'; e.currentTarget.style.boxShadow = '0 20px 50px rgba(0,0,0,0.14)' }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = border; e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}>
                            <div style={{ position: 'absolute', top: -8, right: 12, fontSize: '5rem', fontWeight: 800, color: m.color, opacity: 0.05, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1, userSelect: 'none' }}>{m.num}</div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
                                <div style={{ width: 40, height: 40, borderRadius: 10, background: `${m.color}15`, border: `1px solid ${m.color}35`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <span style={{ fontSize: '0.78rem', fontWeight: 800, color: m.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{m.coef}</span>
                                </div>
                                <div style={{ padding: '0.25rem 0.65rem', borderRadius: 20, background: `${m.color}18`, border: `1px solid ${m.color}30`, fontSize: '0.7rem', fontWeight: 700, color: m.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{m.weight}</div>
                            </div>
                            <div style={{ fontSize: '0.62rem', color: m.color, letterSpacing: '0.1em', textTransform: 'uppercase', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, marginBottom: '0.4rem' }}>{m.sub}</div>
                            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: textH, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>{m.title}</h3>
                            <p style={{ fontSize: '0.8rem', color: textB, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.7 }}>{m.desc}</p>
                            <div style={{ marginTop: '1.25rem' }}>
                                <div style={{ height: 3, background: dark ? '#1E2D45' : '#E5E7EB', borderRadius: 2, overflow: 'hidden' }}>
                                    <div style={{ height: '100%', width: m.weight, background: `linear-gradient(90deg,${m.color},${m.color}99)`, borderRadius: 2 }} />
                                </div>
                                <div style={{ fontSize: '0.65rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.3rem' }}>Poids dans le score final</div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ WORKFLOW ══════════ */}
            <section style={{ padding: '8rem clamp(1.5rem,5vw,4.5rem)', background: dark ? `linear-gradient(180deg,#0A1628 0%,${bg} 100%)` : 'linear-gradient(180deg,#EEF2FF 0%,#F7F8FC 100%)' }}>
                <div style={{ maxWidth: 900, margin: '0 auto' }}>
                    <div style={{ textAlign: 'center', marginBottom: '5rem' }}>
                        <div data-reveal="wf-label" style={{ ...reveal('wf-label'), fontSize: '0.73rem', color: gold, letterSpacing: '0.12em', textTransform: 'uppercase', fontWeight: 600, marginBottom: '1rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Automatisation complète</div>
                        <h2 data-reveal="wf-h2" style={{ ...reveal('wf-h2'), fontSize: 'clamp(2rem,3.5vw,3rem)', fontWeight: 700, letterSpacing: '-0.025em', lineHeight: 1.1, marginBottom: '1rem', color: textH, transitionDelay: '0.1s' }}>
                            De la soumission à la décision —<br />zéro intervention humaine.
                        </h2>
                        <p data-reveal="wf-p" style={{ ...reveal('wf-p'), color: textB, fontSize: '0.95rem', lineHeight: 1.75, maxWidth: 440, margin: '0 auto', fontFamily: 'Helvetica Neue, Arial, sans-serif', transitionDelay: '0.2s' }}>
                            Pour les cas nets. Les cas ambigus (score 30–69) sont traités par un enquêteur sous 48 h.
                        </p>
                    </div>

                    <div style={{ position: 'relative' }}>
                        <div style={{ position: 'absolute', left: 20, top: 28, bottom: 28, width: 1, background: `linear-gradient(to bottom,${gold},#4A9EFF,#1ABC9C)`, opacity: 0.4 }} />
                        {wfSteps.map((step, i) => {
                            const vid = `wf-${i}`
                            const vis = visibleSections.has(vid)
                            return (
                                <div key={vid} data-reveal={vid} style={{ display: 'flex', gap: '2rem', padding: '1.5rem 0', opacity: vis ? 1 : 0, transform: vis ? 'translateX(0)' : 'translateX(-30px)', transition: `opacity 0.7s ease ${i * 0.12}s,transform 0.7s ease ${i * 0.12}s` }}>
                                    <div style={{ width: 40, height: 40, borderRadius: '50%', flexShrink: 0, background: dark ? '#0D1626' : 'white', border: `1.5px solid ${gold}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800, color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif', zIndex: 1, boxShadow: `0 0 16px rgba(201,168,76,0.2)` }}>
                                        {String(i + 1).padStart(2, '0')}
                                    </div>
                                    <div style={{ paddingTop: '0.5rem' }}>
                                        <div style={{ fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.3rem', color: textH }}>{step.title}</div>
                                        <div style={{ fontSize: '0.84rem', color: textB, lineHeight: 1.65, fontFamily: 'Helvetica Neue, Arial, sans-serif', maxWidth: 520 }}>{step.desc}</div>
                                        <div style={{ fontSize: '0.72rem', color: gold, fontWeight: 600, letterSpacing: '0.05em', marginTop: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{step.time}</div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </section>

            {/* ══════════ FEATURES ══════════ */}
            <section data-reveal="features" style={{ ...reveal('features'), padding: '7rem 2rem', maxWidth: 1100, margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <div style={{ fontSize: '0.7rem', letterSpacing: '0.18em', textTransform: 'uppercase', color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>Plateforme complète</div>
                    <h2 style={{ fontSize: 'clamp(1.8rem,4vw,2.8rem)', fontWeight: 400, color: textH, letterSpacing: '-0.02em' }}>Tout ce dont vous avez <strong>besoin</strong></h2>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: '1rem' }}>
                    {features.map(f => (
                        <div key={f.title} style={{ backgroundColor: bgCard2, borderRadius: 14, padding: '1.5rem', border: `1px solid ${border}`, transition: 'border-color 0.2s,transform 0.2s' }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = f.color; e.currentTarget.style.transform = 'translateY(-3px)' }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = border; e.currentTarget.style.transform = 'translateY(0)' }}>
                            <div style={{ width: 36, height: 36, borderRadius: 8, background: `${f.color}15`, border: `1px solid ${f.color}30`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.9rem' }}>
                                <span style={{ fontSize: '0.78rem', fontWeight: 800, color: f.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{f.num}</span>
                            </div>
                            <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: textH, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>{f.title}</h4>
                            <p style={{ fontSize: '0.75rem', color: textB, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.65 }}>{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ CTA ══════════ */}
            <section data-reveal="cta" style={{ ...reveal('cta'), backgroundColor: dark ? '#0A1628' : navy, padding: '7rem 2rem', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 600, height: 600, borderRadius: '50%', background: 'radial-gradient(circle,rgba(201,168,76,0.06) 0%,transparent 70%)', pointerEvents: 'none' }} />
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', width: 900, height: 900, borderRadius: '50%', border: '1px solid rgba(201,168,76,0.06)', pointerEvents: 'none' }} />
                <div style={{ position: 'relative', zIndex: 1, maxWidth: 620, margin: '0 auto' }}>
                    <div style={{ width: 52, height: 52, borderRadius: 12, background: 'linear-gradient(135deg,#C9A84C,#E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.75rem', fontWeight: 800, color: navy, fontSize: '1.3rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', boxShadow: '0 0 32px rgba(201,168,76,0.35)' }}>F</div>
                    <h2 style={{ fontSize: 'clamp(1.8rem,4vw,3rem)', fontWeight: 400, color: 'white', letterSpacing: '-0.02em', marginBottom: '1rem', lineHeight: 1.2 }}>
                        Votre prochain sinistre est<br /><em style={{ fontStyle: 'normal', color: gold }}>déjà en cours d'analyse.</em>
                    </h2>
                    <p style={{ color: 'rgba(255,255,255,0.45)', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.95rem', marginBottom: '2.5rem', lineHeight: 1.7 }}>Rejoignez la plateforme conçue pour le secteur de l'assurance industrielle algérienne.</p>
                    <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '2.5rem' }}>
                        <button onClick={() => navigate('/login')} style={{ padding: '0.9rem 2.2rem', borderRadius: 10, fontSize: '0.9rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', background: 'linear-gradient(135deg,#C9A84C,#E8C97A)', color: navy, border: 'none', boxShadow: '0 6px 24px rgba(201,168,76,0.35)', transition: 'transform 0.15s,box-shadow 0.15s', letterSpacing: '0.04em' }}
                            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 12px 32px rgba(201,168,76,0.45)' }}
                            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 6px 24px rgba(201,168,76,0.35)' }}>
                            Créer un compte gratuit →
                        </button>
                        <button onClick={() => navigate('/login')} style={{ padding: '0.9rem 2.2rem', borderRadius: 10, fontSize: '0.9rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', background: 'transparent', color: 'rgba(255,255,255,0.75)', border: '1.5px solid rgba(255,255,255,0.2)', letterSpacing: '0.04em', transition: 'border-color 0.2s,color 0.2s' }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.6)'; e.currentTarget.style.color = 'white' }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'; e.currentTarget.style.color = 'rgba(255,255,255,0.75)' }}>
                            Se connecter
                        </button>
                    </div>
                    <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                        {['4 modèles IA', 'Multimodal', 'Temps réel', 'Sécurisé'].map(t => (
                            <div key={t} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                <span style={{ color: gold, fontWeight: 700, fontSize: '0.75rem' }}>—</span>
                                <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{t}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ══════════ FOOTER ══════════ */}
            <footer style={{ backgroundColor: dark ? '#060D1A' : '#0A1628', padding: '1.5rem 2.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <div style={{ width: 24, height: 24, borderRadius: 5, background: 'linear-gradient(135deg,#C9A84C,#E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, color: navy, fontSize: '0.7rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>F</div>
                    <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>© 2026 FraudGuard AI · Université M'Hamed Bougara de Boumerdès</span>
                </div>
                <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Projet de fin d'études — Dr. Yahiatene</span>
            </footer>

        </div>
    )
}