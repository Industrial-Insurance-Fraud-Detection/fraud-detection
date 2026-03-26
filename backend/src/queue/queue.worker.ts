import { Injectable, Logger } from '@nestjs/common';
import { RabbitSubscribe, Nack } from '@golevelup/nestjs-rabbitmq';
import { PrismaService } from '../prisma/prisma.service';
import { NotificationsService } from '../notifications/notifications.service';
import { ConfigService } from '@nestjs/config';
import { ClaimStatus, DecisionType, DecisionOutcome } from '@prisma/client';
import axios from 'axios';

interface AnalysisJobPayload {
  claimId: string;
}

/**
 * AI score weights — must sum to 1.0
 * Anomaly detection carries the most weight as sensor data is the
 * most reliable indicator of real vs fabricated equipment failures.
 */
const WEIGHTS = {
  anomaly: 0.35, // Isolation Forest + LSTM Autoencoder
  classification: 0.25, // XGBoost failure classification
  nlp: 0.20, // Multilingual BERT text analysis
  vision: 0.20, // YOLOv8 + ELA image forensics
};

// fraud score thresholds
const AUTO_APPROVE_THRESHOLD = 30;
const AUTO_REJECT_THRESHOLD = 70;

@Injectable()
export class QueueWorker {
  private readonly logger = new Logger(QueueWorker.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly notifications: NotificationsService,
    private readonly config: ConfigService,
  ) { }

  /**
   * Consumes AI analysis jobs from RabbitMQ.
   * Calls all 4 AI microservices in parallel, computes weighted fraud score,
   * persists the result, and triggers auto-decision or routes to human review.
   *
   * Returns Nack(false) on unrecoverable errors to prevent infinite requeue.
   * The claim is left in ANALYZING status so an admin can investigate.
   */
  @RabbitSubscribe({
    exchange: 'taamine',
    routingKey: 'ai-analysis',
    queue: 'ai-analysis',
  })
  async handleAnalysisJob(payload: AnalysisJobPayload): Promise<void | Nack> {
    const { claimId } = payload;
    this.logger.log(`Processing analysis job for claim: ${claimId}`);

    try {
      // mark claim as ANALYZING immediately
      await this.prisma.claim.update({
        where: { id: claimId },
        data: { status: ClaimStatus.ANALYZING },
      });

      // fetch claim with all related data needed by AI services
      const claim = await this.prisma.claim.findUnique({
        where: { id: claimId },
        include: {
          equipment: true,
          files: true,
          client: {
            select: { id: true, email: true, firstName: true, lastName: true },
          },
        },
      });

      if (!claim) {
        this.logger.error(`Claim ${claimId} not found — discarding job`);
        return new Nack(false);
      }

      // locate uploaded files by type
      const csvFile = claim.files.find((f) => f.fileType === 'CSV');
      const photoFiles = claim.files.filter((f) => f.fileType === 'PHOTO');
      const pdfFile = claim.files.find((f) => f.fileType === 'PDF');

      // ── Anomaly service payload ──────────────────────────────────────────
      const anomalyPayload = {
        claimId,
        csvPath: csvFile?.minioPath || null,
        claimDate: claim.incidentDate.toISOString().split('T')[0], // "YYYY-MM-DD"
      };

      // ── Classification service payload ───────────────────────────────────
      // /classify-failure expects a CSV file upload (multipart/form-data)
      // We pass the MinIO path so the service can fetch it directly
      const classificationPayload = {
        claimId,
        csvPath: csvFile?.minioPath || null,
        equipmentType: claim.equipment.type,
      };

      // ── NLP service payload ──────────────────────────────────────────────
      const nlpPayload = {
        claimId,
        claimDescription: claim.description,
        maintenanceReportText: pdfFile?.minioPath
          ? `PDF available at: ${pdfFile.minioPath}`
          : 'No maintenance report provided.',
      };

      // ── Vision service payload ───────────────────────────────────────────
      // photoPaths must be an array — vision service expects list[str]
      const visionPayload = {
        claimId,
        photoPaths: photoFiles.map((f) => f.minioPath).filter(Boolean),
        declaredDamage: claim.description,
        incidentDate: claim.incidentDate.toISOString(),
      };

      // call all 4 AI services in parallel — allSettled never throws
      const [anomalyRes, classRes, nlpRes, visionRes] = await Promise.allSettled([
        axios.post(
          `${this.config.get('AI_ANOMALY_URL') || 'http://localhost:8001'}/detect-anomalies`,
          anomalyPayload,
          { timeout: 60000 },
        ),
        axios.post(
          `${this.config.get('AI_CLASSIFICATION_URL') || 'http://localhost:8002'}/classify-json`,
          classificationPayload,
          { timeout: 60000 },
        ),
        axios.post(
          `${this.config.get('AI_NLP_URL') || 'http://localhost:8003'}/analyze-text`,
          nlpPayload,
          { timeout: 60000 },
        ),
        axios.post(
          `${this.config.get('AI_VISION_URL') || 'http://localhost:8004'}/analyze`,
          visionPayload,
          { timeout: 60000 },
        ),
      ]);

      // extract scores — fallback to 50 (neutral) if a service fails
      const anomalyScore = this.extractScore(anomalyRes, 'anomaly');
      const classificationScore = this.extractScore(classRes, 'classification');
      const nlpScore = this.extractScore(nlpRes, 'nlp');
      const visionScore = this.extractScore(visionRes, 'vision');

      // compute weighted fraud score 0–100
      const finalScore =
        anomalyScore * WEIGHTS.anomaly +
        classificationScore * WEIGHTS.classification +
        nlpScore * WEIGHTS.nlp +
        visionScore * WEIGHTS.vision;

      const fraudClass =
        finalScore < AUTO_APPROVE_THRESHOLD ? 'LOW'
          : finalScore < AUTO_REJECT_THRESHOLD ? 'MEDIUM'
            : 'HIGH';

      // build full breakdown for investigator UI and n8n PDF
      const breakdown = {
        anomaly: anomalyRes.status === 'fulfilled'
          ? anomalyRes.value.data
          : { error: true, message: 'Service unavailable' },
        classification: classRes.status === 'fulfilled'
          ? classRes.value.data
          : { error: true, message: 'Service unavailable' },
        nlp: nlpRes.status === 'fulfilled'
          ? nlpRes.value.data
          : { error: true, message: 'Service unavailable' },
        vision: visionRes.status === 'fulfilled'
          ? visionRes.value.data
          : { error: true, message: 'Service unavailable' },
      };

      // persist AI analysis result
      await this.prisma.aIAnalysis.create({
        data: {
          claimId,
          anomalyScore,
          classificationScore,
          nlpScore,
          visionScore,
          finalScore,
          fraudClass,
          breakdown,
        },
      });

      this.logger.log(
        `Fraud score for claim ${claim.reference}: ${finalScore.toFixed(1)}/100 (${fraudClass})`,
      );

      // route claim based on fraud score
      if (finalScore < AUTO_APPROVE_THRESHOLD) {
        await this.autoDecide(claim, finalScore, breakdown, DecisionOutcome.APPROVED);
      } else if (finalScore >= AUTO_REJECT_THRESHOLD) {
        await this.autoDecide(claim, finalScore, breakdown, DecisionOutcome.REJECTED);
      } else {
        // score 30–69 — requires human investigator review
        await this.prisma.claim.update({
          where: { id: claimId },
          data: { status: ClaimStatus.HUMAN_REVIEW },
        });

        await this.notifications.create(
          claim.client.id,
          "Sinistre en cours d'examen",
          `Votre sinistre ${claim.reference} nécessite une vérification humaine. Délai estimé: 48h.`,
        );

        this.logger.log(
          `Claim ${claim.reference} routed to HUMAN_REVIEW (score: ${finalScore.toFixed(1)})`,
        );
      }
    } catch (err) {
      this.logger.error(
        `Worker failed for claim ${claimId}: ${err.message}`,
        err.stack,
      );
      return new Nack(false);
    }
  }

  /**
   * Creates an automatic decision record and updates claim status.
   * Sends full breakdown to n8n so the PDF contains all AI detail.
   */
  private async autoDecide(
    claim: any,
    score: number,
    breakdown: any,
    outcome: DecisionOutcome,
  ): Promise<void> {
    const newStatus =
      outcome === DecisionOutcome.APPROVED
        ? ClaimStatus.APPROVED
        : ClaimStatus.REJECTED;

    const notes =
      outcome === DecisionOutcome.APPROVED
        ? `Décision automatique — score de fraude faible: ${score.toFixed(1)}/100. Aucun indicateur de fraude détecté.`
        : `Décision automatique — score de fraude élevé: ${score.toFixed(1)}/100. Indicateurs de fraude détectés par le système IA.`;

    // atomic transaction — decision + status update
    await this.prisma.$transaction([
      this.prisma.decision.create({
        data: {
          claimId: claim.id,
          type: DecisionType.AUTO,
          outcome,
          notes,
        },
      }),
      this.prisma.claim.update({
        where: { id: claim.id },
        data: { status: newStatus },
      }),
    ]);

    const verb = outcome === DecisionOutcome.APPROVED ? 'approuvé' : 'rejeté';

    await this.notifications.create(
      claim.client.id,
      `Sinistre ${verb} automatiquement`,
      `Votre sinistre ${claim.reference} a été ${verb} par le système IA (score: ${score.toFixed(0)}/100).`,
    );

    // trigger n8n webhook with full breakdown for rich PDF generation
    const webhookEvent =
      outcome === DecisionOutcome.APPROVED ? 'auto-approved' : 'auto-rejected';
    const base =
      this.config.get('N8N_WEBHOOK_BASE') || 'http://localhost:5678/webhook';

    axios
      .post(`${base}/${webhookEvent}`, {
        claimId: claim.id,
        reference: claim.reference,
        clientEmail: claim.client.email,
        clientName: `${claim.client.firstName} ${claim.client.lastName}`,
        finalScore: score,
        fraudClass: score < AUTO_APPROVE_THRESHOLD ? 'LOW' : 'HIGH',
        claimedAmount: claim.claimedAmount,
        outcome,
        // full breakdown — used by n8n to build detailed PDF
        scores: {
          anomaly: breakdown.anomaly?.score ?? null,
          classification: breakdown.classification?.fraud_score ?? null,
          nlp: breakdown.nlp?.score ?? null,
          vision: breakdown.vision?.score ?? null,
        },
        indicators: [
          ...(breakdown.anomaly?.fraud_indicator
            ? [breakdown.anomaly.fraud_indicator]
            : []),
          ...(breakdown.classification?.predicted_class
            ? [breakdown.classification.predicted_class]
            : []),
          ...(breakdown.nlp?.flaggedSignals ?? []),
          ...(breakdown.vision?.indicators ?? []),
        ],
        anomalyDetail: {
          preIncidentAnomaly: breakdown.anomaly?.pre_incident_anomaly ?? null,
          fraudIndicator: breakdown.anomaly?.fraud_indicator ?? null,
          anomalies: breakdown.anomaly?.anomalies ?? [],
        },
        visionDetail: {
          manipulation: breakdown.vision?.manipulation ?? null,
          exifIssues: breakdown.vision?.exifIssues ?? [],
          boxes: breakdown.vision?.boxes ?? [],
        },
        nlpDetail: {
          label: breakdown.nlp?.label ?? null,
          flaggedSignals: breakdown.nlp?.flaggedSignals ?? [],
          claimScore: breakdown.nlp?.claimScore ?? null,
          maintenanceScore: breakdown.nlp?.maintenanceScore ?? null,
        },
        classificationDetail: {
          predictedClass: breakdown.classification?.predicted_class ?? null,
          classDistribution: breakdown.classification?.class_distribution ?? null,
        },
      })
      .catch((err) =>
        this.logger.warn(`n8n webhook ${webhookEvent} failed: ${err.message}`),
      );

    this.logger.log(
      `Claim ${claim.reference} AUTO-${outcome} (score: ${score.toFixed(1)})`,
    );
  }

  /**
   * Extracts the fraud score from an AI service response.
   * Returns 50 (neutral) if the service failed or returned an invalid score.
   */
  private extractScore(
    result: PromiseSettledResult<any>,
    service: string,
  ): number {
    if (result.status === 'fulfilled') {
      // anomaly service returns { score }
      // classification service returns { fraud_score }
      // nlp service returns { score }
      // vision service returns { score }
      const data = result.value?.data;
      const score = data?.score ?? data?.fraud_score;
      if (typeof score === 'number' && score >= 0 && score <= 100) {
        return score;
      }
      this.logger.warn(
        `AI service "${service}" returned invalid score: ${score} — defaulting to 50`,
      );
    } else {
      this.logger.warn(
        `AI service "${service}" failed: ${result.reason?.message} — defaulting to 50`,
      );
    }
    return 50;
  }
}