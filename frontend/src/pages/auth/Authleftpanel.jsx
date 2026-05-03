// ── Shared Left Panel — used across all 3 auth pages ─────────────────────────
export default function AuthLeftPanel() {
    return (
        <div style={{
            width: '45%',
            background: 'linear-gradient(160deg, #060D1A 0%, #0A1628 45%, #0F2347 100%)',
            display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
            padding: '3rem 2.5rem', position: 'relative', overflow: 'hidden', minHeight: '100vh',
        }}>
            <style>{`
        @keyframes rotateRing  { from { transform: rotateZ(0deg) }   to { transform: rotateZ(360deg) } }
        @keyframes rotateRingR { from { transform: rotateZ(0deg) }   to { transform: rotateZ(-360deg) } }
        @keyframes floatCore   { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-14px) scale(1.03)} }
        @keyframes orb1  { from{transform:rotateZ(0deg)   translateX(80px)  rotateZ(0deg)}   to{transform:rotateZ(360deg)   translateX(80px)  rotateZ(-360deg)} }
        @keyframes orb2  { from{transform:rotateZ(120deg) translateX(110px) rotateZ(-120deg)} to{transform:rotateZ(480deg)  translateX(110px) rotateZ(-480deg)} }
        @keyframes orb3  { from{transform:rotateZ(240deg) translateX(148px) rotateZ(-240deg)} to{transform:rotateZ(600deg)  translateX(148px) rotateZ(-600deg)} }
        @keyframes orb4  { from{transform:rotateZ(60deg)  translateX(148px) rotateZ(-60deg)}  to{transform:rotateZ(420deg)  translateX(148px) rotateZ(-420deg)} }
        @keyframes pulseGlow { 0%,100%{opacity:0.4;transform:scale(1)} 50%{opacity:0.75;transform:scale(1.12)} }
        @keyframes fadeGrid  { 0%,100%{opacity:0.03} 50%{opacity:0.07} }
        @keyframes slideUp   { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
        @keyframes particleFloat { 0%{transform:translateY(0) translateX(0);opacity:0} 10%{opacity:1} 90%{opacity:0.6} 100%{transform:translateY(-120px) translateX(20px);opacity:0} }
        @keyframes scanDown  { 0%{top:-60px} 100%{top:110%} }
        @keyframes dashMove  { from{stroke-dashoffset:400} to{stroke-dashoffset:0} }
      `}</style>

            {/* Grid background */}
            <div style={{
                position: 'absolute', inset: 0, pointerEvents: 'none',
                backgroundImage: `
          linear-gradient(rgba(201,168,76,0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(201,168,76,0.05) 1px, transparent 1px)
        `,
                backgroundSize: '48px 48px',
                animation: 'fadeGrid 6s ease-in-out infinite',
            }} />

            {/* Scan line */}
            <div style={{
                position: 'absolute', left: 0, right: 0, height: 60,
                background: 'linear-gradient(transparent, rgba(201,168,76,0.04), transparent)',
                animation: 'scanDown 8s linear infinite',
                pointerEvents: 'none', zIndex: 1,
            }} />

            {/* Corner brackets */}
            {[
                { top: 16, left: 16, borderTop: '2px solid rgba(201,168,76,0.5)', borderLeft: '2px solid rgba(201,168,76,0.5)' },
                { top: 16, right: 16, borderTop: '2px solid rgba(201,168,76,0.5)', borderRight: '2px solid rgba(201,168,76,0.5)' },
                { bottom: 16, left: 16, borderBottom: '2px solid rgba(201,168,76,0.5)', borderLeft: '2px solid rgba(201,168,76,0.5)' },
                { bottom: 16, right: 16, borderBottom: '2px solid rgba(201,168,76,0.5)', borderRight: '2px solid rgba(201,168,76,0.5)' },
            ].map((s, i) => (
                <div key={i} style={{ position: 'absolute', width: 22, height: 22, ...s, pointerEvents: 'none' }} />
            ))}

            {/* ── 3D Orbit Scene ── */}
            <div style={{
                position: 'relative', width: 320, height: 320,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: '2.5rem', zIndex: 2,
            }}>

                {/* Outer glow */}
                <div style={{
                    position: 'absolute', width: 300, height: 300, borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(201,168,76,0.06) 0%, transparent 70%)',
                    animation: 'pulseGlow 4s ease-in-out infinite',
                }} />

                {/* Ring 1 — tilted 20deg */}
                <div style={{
                    position: 'absolute', width: 160, height: 160, borderRadius: '50%',
                    border: '1.5px solid rgba(201,168,76,0.25)',
                    transform: 'rotateX(70deg)',
                    animation: 'rotateRing 8s linear infinite',
                    boxShadow: '0 0 20px rgba(201,168,76,0.05)',
                }}>
                    {/* Dot on ring 1 */}
                    <div style={{
                        position: 'absolute', top: -5, left: '50%', transform: 'translateX(-50%)',
                        width: 10, height: 10, borderRadius: '50%',
                        background: '#C9A84C', boxShadow: '0 0 14px #C9A84C, 0 0 28px rgba(201,168,76,0.5)',
                    }} />
                </div>

                {/* Ring 2 — tilted 50deg */}
                <div style={{
                    position: 'absolute', width: 220, height: 220, borderRadius: '50%',
                    border: '1px solid rgba(100,160,255,0.2)',
                    transform: 'rotateX(60deg) rotateZ(40deg)',
                    animation: 'rotateRingR 12s linear infinite',
                }}>
                    <div style={{
                        position: 'absolute', top: -5, left: '50%', transform: 'translateX(-50%)',
                        width: 8, height: 8, borderRadius: '50%',
                        background: '#64A0FF', boxShadow: '0 0 12px #64A0FF, 0 0 24px rgba(100,160,255,0.5)',
                    }} />
                </div>

                {/* Ring 3 — tilted 80deg */}
                <div style={{
                    position: 'absolute', width: 296, height: 296, borderRadius: '50%',
                    border: '1px dashed rgba(201,168,76,0.12)',
                    transform: 'rotateX(80deg) rotateZ(-20deg)',
                    animation: 'rotateRing 18s linear infinite',
                }}>
                    <div style={{
                        position: 'absolute', top: -4, left: '50%', transform: 'translateX(-50%)',
                        width: 7, height: 7, borderRadius: '50%',
                        background: 'rgba(201,168,76,0.7)', boxShadow: '0 0 10px rgba(201,168,76,0.5)',
                    }} />
                </div>

                {/* ── Central shield core ── */}
                <div style={{ animation: 'floatCore 6s ease-in-out infinite', zIndex: 5 }}>
                    {/* Outer glow ring */}
                    <div style={{
                        width: 130, height: 130, borderRadius: '50%',
                        border: '1px solid rgba(201,168,76,0.3)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        animation: 'pulseGlow 3s ease-in-out infinite',
                    }}>
                        {/* Core sphere */}
                        <div style={{
                            width: 100, height: 100, borderRadius: '50%',
                            background: 'linear-gradient(135deg, rgba(201,168,76,0.25) 0%, rgba(10,22,40,0.9) 60%, rgba(15,35,71,0.95) 100%)',
                            border: '2px solid rgba(201,168,76,0.45)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            boxShadow: '0 0 30px rgba(201,168,76,0.15), 0 0 60px rgba(10,22,40,0.8), inset 0 1px 0 rgba(255,255,255,0.1)',
                        }}>
                            <span style={{ fontSize: '2.8rem', filter: 'drop-shadow(0 0 12px rgba(201,168,76,0.7))' }}>🛡</span>
                        </div>
                    </div>
                    {/* Soft reflection */}
                    <div style={{
                        width: 100, height: 28, margin: '6px auto 0',
                        background: 'linear-gradient(rgba(201,168,76,0.08), transparent)',
                        borderRadius: '0 0 50% 50%', filter: 'blur(6px)',
                    }} />
                </div>

                {/* Floating micro-cards */}
                {[
                    { style: { top: 28, left: 18 }, icon: '📡', label: 'Anomalie', val: '35%', color: '#E67E22', delay: '0s', anim: 'floatCore 7s ease-in-out infinite' },
                    { style: { top: 28, right: 18 }, icon: '📝', label: 'NLP', val: '20%', color: '#C9A84C', delay: '1.8s', anim: 'floatCore 8s ease-in-out infinite' },
                    { style: { bottom: 44, left: 12 }, icon: '👁', label: 'Vision', val: '20%', color: '#1A7A4A', delay: '3.5s', anim: 'floatCore 6.5s ease-in-out infinite' },
                    { style: { bottom: 44, right: 12 }, icon: '⚡', label: 'Classif.', val: '25%', color: '#1A5276', delay: '5s', anim: 'floatCore 9s ease-in-out infinite' },
                ].map((c, i) => (
                    <div key={i} style={{
                        position: 'absolute', ...c.style,
                        background: 'rgba(6,13,26,0.88)',
                        border: '1px solid rgba(201,168,76,0.22)',
                        borderRadius: 10, padding: '0.45rem 0.7rem',
                        backdropFilter: 'blur(12px)',
                        animation: c.anim, animationDelay: c.delay,
                        boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
                        zIndex: 6,
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', marginBottom: 2 }}>
                            <span style={{ fontSize: '0.7rem' }}>{c.icon}</span>
                            <span style={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.65rem', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>{c.label}</span>
                        </div>
                        <div style={{ color: c.color, fontWeight: 700, fontSize: '0.9rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', lineHeight: 1 }}>{c.val}</div>
                    </div>
                ))}

                {/* Floating particles */}
                {[
                    { left: '20%', bottom: '15%', size: 3, delay: '0s', dur: '4s' },
                    { left: '75%', bottom: '20%', size: 2, delay: '1.5s', dur: '5s' },
                    { left: '45%', bottom: '10%', size: 4, delay: '2.8s', dur: '3.5s' },
                    { left: '60%', bottom: '30%', size: 2, delay: '4s', dur: '6s' },
                ].map((p, i) => (
                    <div key={i} style={{
                        position: 'absolute', left: p.left, bottom: p.bottom,
                        width: p.size, height: p.size, borderRadius: '50%',
                        background: '#C9A84C',
                        animation: `particleFloat ${p.dur} ease-in-out infinite`,
                        animationDelay: p.delay,
                        opacity: 0,
                    }} />
                ))}
            </div>

            {/* ── Text block ── */}
            <div style={{ textAlign: 'center', zIndex: 2, maxWidth: 320, animation: 'slideUp 0.8s ease 0.2s both' }}>

                {/* Brand */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.65rem', marginBottom: '1.25rem' }}>
                    <div style={{
                        width: 38, height: 38, borderRadius: 9,
                        background: 'linear-gradient(135deg, #C9A84C, #E8C97A)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '1.1rem', boxShadow: '0 0 20px rgba(201,168,76,0.35)',
                    }}>🛡</div>
                    <div style={{ textAlign: 'left' }}>
                        <div style={{ color: 'white', fontWeight: 700, fontSize: '1rem', letterSpacing: '0.04em', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>FraudGuard AI</div>
                        <div style={{ color: '#C9A84C', fontSize: '0.56rem', letterSpacing: '0.22em', textTransform: 'uppercase', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginTop: 1 }}>Industrial Insurance</div>
                    </div>
                </div>

                {/* Divider */}
                <div style={{ width: 40, height: 1, background: 'linear-gradient(90deg, transparent, #C9A84C, transparent)', margin: '0 auto 1.25rem' }} />

                {/* Headline */}
                <h1 style={{ color: 'white', fontSize: '1.55rem', fontWeight: 400, lineHeight: 1.4, marginBottom: '0.85rem', fontFamily: 'Georgia,serif', letterSpacing: '-0.01em' }}>
                    Détection de fraude<br />
                    <span style={{ color: '#C9A84C', fontStyle: 'italic' }}>industrielle</span> par IA
                </h1>

                <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.78rem', lineHeight: 1.8, fontFamily: 'Helvetica Neue,Arial,sans-serif', marginBottom: '1.5rem' }}>
                    Analyse multimodale en temps réel —<br />capteurs, photos & rapports techniques.
                </p>

                {/* 4 model pills */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', justifyContent: 'center' }}>
                    {[
                        ['📡', 'LSTM + Isolation Forest'],
                        ['⚡', 'XGBoost Classifier'],
                        ['📝', 'BERT NLP'],
                        ['👁', 'YOLOv8 Vision'],
                    ].map(([icon, label]) => (
                        <div key={label} style={{
                            display: 'flex', alignItems: 'center', gap: '0.35rem',
                            padding: '0.3rem 0.7rem',
                            background: 'rgba(201,168,76,0.08)',
                            border: '1px solid rgba(201,168,76,0.18)',
                            borderRadius: 20,
                        }}>
                            <span style={{ fontSize: '0.7rem' }}>{icon}</span>
                            <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.65rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', letterSpacing: '0.04em' }}>{label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div style={{
                position: 'absolute', bottom: 20,
                color: 'rgba(255,255,255,0.18)', fontSize: '0.62rem',
                fontFamily: 'Helvetica Neue,Arial,sans-serif', textAlign: 'center', zIndex: 2,
                letterSpacing: '0.04em',
            }}>
                © 2026 FraudGuard AI · Université M'Hamed Bougara · Boumerdès
            </div>
        </div>
    )
}