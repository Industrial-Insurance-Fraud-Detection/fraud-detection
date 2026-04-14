import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'

const COLORS = {
  primary:   [15, 35, 71],    // #0F2347
  gold:      [201, 168, 76],  // #C9A84C
  green:     [26, 122, 74],   // #1A7A4A
  red:       [192, 57, 43],   // #C0392B
  orange:    [243, 156, 18],  // #F39C12
  lightGray: [247, 248, 252], // #F7F8FC
  gray:      [107, 114, 128], // #6B7280
  darkGray:  [75, 85, 99],    // #4B5563
  white:     [255, 255, 255],
}

const STATUS_CONFIG = {
  APPROVED:     { label: 'APPROUVE',         color: COLORS.green },
  REJECTED:     { label: 'REJETE',           color: COLORS.red },
  PENDING:      { label: 'EN ATTENTE',       color: COLORS.orange },
  ANALYZING:    { label: 'ANALYSE EN COURS', color: [26, 82, 118] },
  HUMAN_REVIEW: { label: 'REVISION HUMAINE', color: [26, 82, 118] },
}

const FRAUD_LABELS = {
  'NO PRECURSOR DETECTED':     'Aucun precurseur detecte',
  'GRADUAL DEGRADATION':       'Degradation progressive',
  'ABRUPT FAILURE':            'Panne abrupte',
  'MINOR PRECURSORS DETECTED': 'Precurseurs mineurs detectes',
  'UNKNOWN':                   'Inconnu',
}

export function exportClaimPDF(claim, user) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
  const pageW = doc.internal.pageSize.getWidth()
  const pageH = doc.internal.pageSize.getHeight()
  let y = 0

  // ─────────────────────────────────────────
  // HEADER
  // ─────────────────────────────────────────
  doc.setFillColor(...COLORS.primary)
  doc.rect(0, 0, pageW, 45, 'F')

  // Logo texte
  doc.setTextColor(...COLORS.gold)
  doc.setFontSize(18)
  doc.setFont('helvetica', 'bold')
  doc.text('FraudGuard AI', 15, 18)

  doc.setTextColor(...COLORS.white)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'normal')
  doc.text('INDUSTRIAL INSURANCE', 15, 24)

  // Titre rapport
  doc.setTextColor(...COLORS.white)
  doc.setFontSize(14)
  doc.setFont('helvetica', 'bold')
  doc.text('RAPPORT D\'ANALYSE DE SINISTRE', pageW / 2, 18, { align: 'center' })

  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.text(`Genere le ${new Date().toLocaleDateString('fr-FR', { day: '2-digit', month: 'long', year: 'numeric' })} a ${new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`, pageW / 2, 26, { align: 'center' })

  // Reference en haut à droite
  doc.setFontSize(10)
  doc.setFont('helvetica', 'bold')
  doc.text(claim.reference || 'N/A', pageW - 15, 18, { align: 'right' })
  doc.setFontSize(7)
  doc.setFont('helvetica', 'normal')
  doc.text('REFERENCE', pageW - 15, 24, { align: 'right' })

  // Bande dorée
  doc.setFillColor(...COLORS.gold)
  doc.rect(0, 45, pageW, 1.5, 'F')

  y = 55

  // ─────────────────────────────────────────
  // STATUT BADGE
  // ─────────────────────────────────────────
  const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
  doc.setFillColor(...sc.color)
  doc.roundedRect(15, y - 5, 50, 10, 2, 2, 'F')
  doc.setTextColor(...COLORS.white)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.text(sc.label, 40, y + 1, { align: 'center' })

  // Score
  const score = claim.finalScore !== null && claim.finalScore !== undefined ? Math.round(claim.finalScore) : null
  if (score !== null) {
    const scoreColor = score > 70 ? COLORS.red : score > 30 ? COLORS.orange : COLORS.green
    doc.setFillColor(...scoreColor)
    doc.roundedRect(pageW - 65, y - 5, 50, 10, 2, 2, 'F')
    doc.setTextColor(...COLORS.white)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'bold')
    doc.text(`SCORE IA : ${score}/100`, pageW - 40, y + 1, { align: 'center' })
  }

  y += 15

  // ─────────────────────────────────────────
  // SECTION 1 — INFORMATIONS DU SINISTRE
  // ─────────────────────────────────────────
  doc.setFillColor(...COLORS.primary)
  doc.rect(15, y, pageW - 30, 8, 'F')
  doc.setTextColor(...COLORS.white)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'bold')
  doc.text('1. INFORMATIONS DU SINISTRE', 20, y + 5.5)
  y += 12

  const infoData = [
    ['Equipement',     claim.equipment || '-'],
    ['Date incident',  claim.incidentDate ? new Date(claim.incidentDate).toLocaleDateString('fr-FR') : '-'],
    ['Lieu',           claim.location || 'Non specifie'],
    ['Montant reclame',claim.amount ? `${claim.amount.toLocaleString('fr-FR')} DA` : '-'],
    ['Date soumission',claim.createdAt ? new Date(claim.createdAt).toLocaleDateString('fr-FR') : '-'],
    ['Client',         user?.fullName || '-'],
    ['Entreprise',     user?.company || 'Non specifie'],
  ]

  autoTable(doc, {
    startY: y,
    head: [],
    body: infoData,
    margin: { left: 15, right: 15 },
    styles: { fontSize: 9, cellPadding: 3, fontStyle: 'normal' },
    columnStyles: {
      0: { fontStyle: 'bold', cellWidth: 50, fillColor: COLORS.lightGray, textColor: COLORS.darkGray },
      1: { textColor: COLORS.primary },
    },
    theme: 'plain',
    tableLineColor: [229, 231, 235],
    tableLineWidth: 0.1,
  })

  y = doc.lastAutoTable.finalY + 8

  // Description
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...COLORS.darkGray)
  doc.text('Description :', 15, y)
  y += 5

  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...COLORS.darkGray)
  const descLines = doc.splitTextToSize(claim.description || 'Aucune description', pageW - 30)
  doc.text(descLines, 15, y)
  y += descLines.length * 5 + 8

  // ─────────────────────────────────────────
  // SECTION 2 — ANALYSE IA
  // ─────────────────────────────────────────
  if (y > pageH - 80) { doc.addPage(); y = 20 }

  doc.setFillColor(...COLORS.primary)
  doc.rect(15, y, pageW - 30, 8, 'F')
  doc.setTextColor(...COLORS.white)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'bold')
  doc.text('2. RESULTATS DE L\'ANALYSE IA', 20, y + 5.5)
  y += 12

  if (score !== null) {
    const aiData = [
      ['Modele 1 — Anomalie capteurs (35%)',  claim.anomalyScore !== null ? `${Math.round(claim.anomalyScore)}/100` : 'N/A'],
      ['Modele 2 — Classification panne (25%)', claim.classificationScore !== null ? `${Math.round(claim.classificationScore)}/100` : 'N/A'],
      ['Modele 3 — Analyse NLP (20%)',          claim.nlpScore !== null ? `${Math.round(claim.nlpScore)}/100` : 'N/A'],
      ['Modele 4 — Vision photos (20%)',         claim.visionScore !== null ? `${Math.round(claim.visionScore)}/100` : 'N/A'],
      ['SCORE FINAL COMBINE',                   `${score}/100`],
    ]

    autoTable(doc, {
      startY: y,
      head: [['Modele IA', 'Score']],
      body: aiData,
      margin: { left: 15, right: 15 },
      headStyles: { fillColor: COLORS.gold, textColor: COLORS.primary, fontStyle: 'bold', fontSize: 8 },
      styles: { fontSize: 9, cellPadding: 3 },
      columnStyles: {
        0: { cellWidth: 130 },
        1: { halign: 'center', fontStyle: 'bold' },
      },
      bodyStyles: { textColor: COLORS.darkGray },
      didParseCell: (data) => {
        if (data.row.index === 4) {
          data.cell.styles.fillColor = COLORS.primary
          data.cell.styles.textColor = COLORS.white
          data.cell.styles.fontStyle = 'bold'
        }
      },
      theme: 'striped',
      alternateRowStyles: { fillColor: COLORS.lightGray },
    })

    y = doc.lastAutoTable.finalY + 8

    // Indicateur fraude
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...COLORS.darkGray)
    doc.text('Indicateur de fraude :', 15, y)
    doc.setFont('helvetica', 'normal')
    doc.text(FRAUD_LABELS[claim.fraudIndicator] || claim.fraudIndicator || 'N/A', 70, y)
    y += 6

    doc.text('Precurseurs avant incident :', 15, y)
    doc.setFont('helvetica', 'bold')
    const color = claim.preIncidentAnomaly ? [26, 122, 74] : [192, 57, 43]
    doc.setTextColor(...color)
    doc.text(claim.preIncidentAnomaly ? 'Anomalies detectees' : 'Aucun precurseur', 70, y)
    y += 10
  } else {
    doc.setFontSize(9)
    doc.setTextColor(...COLORS.gray)
    doc.text('Analyse IA en cours...', 15, y)
    y += 10
  }

  // ─────────────────────────────────────────
  // SECTION 3 — DECISION
  // ─────────────────────────────────────────
  if (y > pageH - 60) { doc.addPage(); y = 20 }

  doc.setFillColor(...COLORS.primary)
  doc.rect(15, y, pageW - 30, 8, 'F')
  doc.setTextColor(...COLORS.white)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'bold')
  doc.text('3. DECISION FINALE', 20, y + 5.5)
  y += 12

  if (claim.investigatorNote) {
    const decisionColor = claim.status === 'APPROVED' ? COLORS.green : COLORS.red
    doc.setFillColor(...decisionColor)
    doc.roundedRect(15, y, pageW - 30, 8, 2, 2, 'F')
    doc.setTextColor(...COLORS.white)
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.text(`DECISION : ${sc.label}`, pageW / 2, y + 5.5, { align: 'center' })
    y += 12

    doc.setFontSize(8)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...COLORS.darkGray)
    doc.text('Commentaire de l\'investigateur :', 15, y)
    y += 5
    doc.setFont('helvetica', 'normal')
    const noteLines = doc.splitTextToSize(claim.investigatorNote, pageW - 30)
    doc.text(noteLines, 15, y)
    y += noteLines.length * 5 + 5

    if (claim.decidedAt) {
      doc.setFontSize(8)
      doc.setTextColor(...COLORS.gray)
      doc.text(`Decision prise le ${new Date(claim.decidedAt).toLocaleString('fr-FR')}`, 15, y)
      y += 8
    }
  } else {
    doc.setFontSize(9)
    doc.setTextColor(...COLORS.gray)
    doc.text('Aucune decision prise pour le moment.', 15, y)
    y += 10
  }

  // ─────────────────────────────────────────
  // FOOTER
  // ─────────────────────────────────────────
  const totalPages = doc.internal.getNumberOfPages()
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i)
    doc.setFillColor(...COLORS.primary)
    doc.rect(0, pageH - 15, pageW, 15, 'F')
    doc.setFillColor(...COLORS.gold)
    doc.rect(0, pageH - 16, pageW, 1, 'F')
    doc.setTextColor(...COLORS.white)
    doc.setFontSize(7)
    doc.setFont('helvetica', 'normal')
    doc.text('FraudGuard AI — Universite M\'Hamed Bougara de Boumerdes — Document confidentiel', pageW / 2, pageH - 8, { align: 'center' })
    doc.text(`Page ${i} / ${totalPages}`, pageW - 15, pageH - 8, { align: 'right' })
  }

  // Télécharger
  doc.save(`${claim.reference || 'rapport'}-fraudguard.pdf`)
}