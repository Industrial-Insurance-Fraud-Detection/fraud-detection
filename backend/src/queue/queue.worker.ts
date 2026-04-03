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
 */
const WEIGHTS = {
  anomaly: 0.35,
  classification: 0.25,
  nlp: 0.20,
  vision: 0.20,
};

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

  @RabbitSubscribe({
    exchange: 'taamine',
    routingKey: 'ai-analysis',
    queue: 'ai-analysis',
  })
  async handleAnalysisJob(payload: AnalysisJobPayload): Promise<void | Nack> {
    const { claimId } = payload;
    this.logger.log(`Processing analysis job for claim: ${claimId}`);

    try {
      await this.prisma.claim.update({
        where: { id: claimId },
        data: { status: ClaimStatus.ANALYZING },
      });

      const claim = await this.prisma.claim.findUnique({
        where: { id: claimId },
        include: {
          equipment: true,
          files: true,
          client: {
            select: {
              id: true,
              email: true,
              firstName: true,
              lastName: true,
              phone: true,
              wilaya: true,
              company: true,
            },
          },
        },
      });

      if (!claim) {
        this.logger.error(`Claim ${claimId} not found — discarding job`);
        return new Nack(false);
      }

      const csvFile = claim.files.find((f) => f.fileType === 'CSV');
      const photoFiles = claim.files.filter((f) => f.fileType === 'PHOTO');
      const pdfFile = claim.files.find((f) => f.fileType === 'PDF');

      const anomalyPayload = {
        claimId,
        csvPath: csvFile?.minioPath || null,
        claimDate: claim.incidentDate.toISOString().split('T')[0],
      };

      const classificationPayload = {
        claimId,
        csvPath: csvFile?.minioPath || null,
        equipmentType: claim.equipment.type,
      };

      const nlpPayload = {
        claimId,
        claimDescription: claim.description,
        maintenanceReportText: pdfFile?.minioPath
          ? `PDF available at: ${pdfFile.minioPath}`
          : 'No maintenance report provided.',
      };

      const visionPayload = {
        claimId,
        photoPaths: photoFiles.map((f) => f.minioPath).filter(Boolean),
        declaredDamage: claim.description,
        incidentDate: claim.incidentDate.toISOString(),
      };

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

      const anomalyScore = this.extractScore(anomalyRes, 'anomaly');
      const classificationScore = this.extractScore(classRes, 'classification');
      const nlpScore = this.extractScore(nlpRes, 'nlp');
      const visionScore = this.extractScore(visionRes, 'vision');

      const finalScore =
        anomalyScore * WEIGHTS.anomaly +
        classificationScore * WEIGHTS.classification +
        nlpScore * WEIGHTS.nlp +
        visionScore * WEIGHTS.vision;

      const fraudClass =
        finalScore < AUTO_APPROVE_THRESHOLD ? 'LOW'
          : finalScore < AUTO_REJECT_THRESHOLD ? 'MEDIUM'
            : 'HIGH';

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

      if (finalScore < AUTO_APPROVE_THRESHOLD) {
        await this.autoDecide(
          claim,
          finalScore,
          fraudClass,
          breakdown,
          anomalyScore,
          classificationScore,
          nlpScore,
          visionScore,
          DecisionOutcome.APPROVED,
        );
      } else if (finalScore >= AUTO_REJECT_THRESHOLD) {
        await this.autoDecide(
          claim,
          finalScore,
          fraudClass,
          breakdown,
          anomalyScore,
          classificationScore,
          nlpScore,
          visionScore,
          DecisionOutcome.REJECTED,
        );
      } else {
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
   * Creates an automatic decision and fires the n8n webhook.
   * Uses the same /webhook/human-decision endpoint as investigator decisions
   * so one n8n workflow handles all three cases (auto-approve, auto-reject, human).
   */
  private async autoDecide(
    claim: any,
    score: number,
    fraudClass: string,
    breakdown: any,
    anomalyScore: number,
    classificationScore: number,
    nlpScore: number,
    visionScore: number,
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

    /*
     * Fire webhook to /human-decision — same endpoint used by investigator decisions.
     * The n8n If node checks outcome === APPROVED to branch into correct PDF template.
     * This way one single n8n workflow handles all three decision cases.
     */
    const base =
      this.config.get('N8N_WEBHOOK_BASE') || 'http://localhost:5678/webhook';

    axios
      .post(`${base}/human-decision`, {
        claimId: claim.id,
        reference: claim.reference,
        clientId: claim.client.id,
        clientEmail: claim.client.email,
        clientName: `${claim.client.firstName} ${claim.client.lastName}`,
        clientPhone: claim.client.phone ?? 'N/A',
        clientWilaya: claim.client.wilaya ?? 'N/A',
        clientCompany: claim.client.company ?? 'N/A',
        equipmentName: claim.equipment.name,
        equipmentType: claim.equipment.type,
        incidentDate: claim.incidentDate,
        description: claim.description,
        claimedAmount: claim.claimedAmount,
        outcome,
        notes,
        decidedAt: new Date().toISOString(),
        finalScore: score,
        fraudClass,
        anomalyScore,
        classificationScore,
        nlpScore,
        visionScore,
      })
      .catch((err) =>
        this.logger.warn(`n8n webhook human-decision failed: ${err.message}`),
      );

    this.logger.log(
      `Claim ${claim.reference} AUTO-${outcome} (score: ${score.toFixed(1)})`,
    );
  }

  private extractScore(
    result: PromiseSettledResult<any>,
    service: string,
  ): number {
    if (result.status === 'fulfilled') {
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