import {
  Injectable,
  NotFoundException,
  ForbiddenException,
  BadRequestException,
  Logger,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { MinioService } from '../files/minio.service';
import { QueueProducer } from '../queue/queue.producer';
import { NotificationsService } from '../notifications/notifications.service';
import { EquipmentService } from '../equipment/equipment.service';
import { CreateClaimDto } from './dto/create-claim.dto';
import { DecideClaimDto } from './dto/decide-claim.dto';
import { DecisionOutcome, DecisionType, FileType, ClaimStatus } from '@prisma/client';
import { PaginationDto, paginate } from '../common/dto/pagination.dto';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { ConfigService } from '@nestjs/config';

const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_FILES = 20;
const MAX_CSV_FILES = 3;
const MAX_PHOTO_FILES = 10;

@Injectable()
export class ClaimsService {
  private readonly logger = new Logger(ClaimsService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly minio: MinioService,
    private readonly queueProducer: QueueProducer,
    private readonly notifications: NotificationsService,
    private readonly equipmentService: EquipmentService,
    private readonly config: ConfigService,
  ) { }

  async submitClaim(
    clientId: string,
    dto: CreateClaimDto,
    files: Express.Multer.File[],
  ) {
    const incidentDate = new Date(dto.incidentDate);
    if (incidentDate > new Date()) {
      throw new BadRequestException('incidentDate cannot be in the future');
    }

    await this.equipmentService.verifyActiveAndOwned(dto.equipmentId, clientId);

    if (!files || files.length === 0) {
      throw new BadRequestException('At least one file is required');
    }
    if (files.length > MAX_FILES) {
      throw new BadRequestException(`Cannot upload more than ${MAX_FILES} files`);
    }

    const csvFiles = files.filter(
      (f) => f.mimetype === 'text/csv' || f.originalname.endsWith('.csv'),
    );
    const photoFiles = files.filter((f) => f.mimetype.startsWith('image/'));
    const pdfFiles = files.filter(
      (f) => f.mimetype === 'application/pdf' || f.originalname.endsWith('.pdf'),
    );

    if (csvFiles.length === 0) {
      throw new BadRequestException('At least one CSV sensor data file is required');
    }
    if (photoFiles.length === 0) {
      throw new BadRequestException('At least one equipment photo is required');
    }
    if (csvFiles.length > MAX_CSV_FILES) {
      throw new BadRequestException(`Cannot upload more than ${MAX_CSV_FILES} CSV files`);
    }
    if (photoFiles.length > MAX_PHOTO_FILES) {
      throw new BadRequestException(`Cannot upload more than ${MAX_PHOTO_FILES} photos`);
    }

    for (const file of files) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        throw new BadRequestException(
          `File "${file.originalname}" exceeds the 10MB size limit`,
        );
      }
    }

    const reference = `SIN-${new Date().getFullYear()}-${uuidv4().slice(0, 6).toUpperCase()}`;

    const claim = await this.prisma.claim.create({
      data: {
        reference,
        clientId,
        equipmentId: dto.equipmentId,
        incidentDate,
        description: dto.description,
        claimedAmount: dto.claimedAmount,
        status: ClaimStatus.PENDING,
      },
    });

    for (const file of files) {
      try {
        const minioPath = await this.minio.upload(claim.id, file);
        const fileType = this.detectFileType(file);
        await this.prisma.claimFile.create({
          data: {
            claimId: claim.id,
            fileType,
            minioPath,
            fileName: file.originalname,
            fileSize: file.size,
          },
        });
      } catch (error) {
        this.logger.error(
          `Failed to upload file "${file.originalname}" for claim ${claim.id}`,
          error,
        );
        throw new BadRequestException(
          `Failed to upload file "${file.originalname}". Please try again.`,
        );
      }
    }

    await this.queueProducer.publishAnalysisJob({ claimId: claim.id });
    this.logger.log(`Analysis job queued for claim ${reference}`);

    await this.notifications.create(
      clientId,
      'Sinistre reçu',
      `Votre sinistre ${reference} a été reçu et est en cours d'analyse IA.`,
    );

    this.triggerWebhook('claim-received', {
      claimId: claim.id,
      reference,
      clientId,
    }).catch((err) =>
      this.logger.warn(`n8n webhook claim-received failed: ${err.message}`),
    );

    return {
      claimId: claim.id,
      reference,
      status: ClaimStatus.PENDING,
      filesUploaded: files.length,
      csvCount: csvFiles.length,
      photoCount: photoFiles.length,
      pdfCount: pdfFiles.length,
    };
  }

  async findMyClaims(clientId: string, pagination: PaginationDto) {
    const { page = 1, limit = 10 } = pagination;
    const skip = (page - 1) * limit;

    const [data, total] = await Promise.all([
      this.prisma.claim.findMany({
        where: { clientId },
        skip,
        take: limit,
        orderBy: { createdAt: 'desc' },
        include: {
          equipment: { select: { name: true, type: true } },
          analysis: { select: { finalScore: true, fraudClass: true } },
          decision: { select: { outcome: true, type: true } },
        },
      }),
      this.prisma.claim.count({ where: { clientId } }),
    ]);

    return paginate(data, total, page, limit);
  }

  async getFlaggedClaims(pagination: PaginationDto) {
    const { page = 1, limit = 10 } = pagination;
    const skip = (page - 1) * limit;

    const where = { status: ClaimStatus.HUMAN_REVIEW };

    const [data, total] = await Promise.all([
      this.prisma.claim.findMany({
        where,
        skip,
        take: limit,
        orderBy: { analysis: { finalScore: 'desc' } },
        include: {
          equipment: { select: { name: true, type: true } },
          client: {
            select: { firstName: true, lastName: true, company: true },
          },
          analysis: {
            select: { finalScore: true, fraudClass: true, analyzedAt: true },
          },
        },
      }),
      this.prisma.claim.count({ where }),
    ]);

    return paginate(data, total, page, limit);
  }

  async findOne(id: string, userId: string, userRole: string) {
    const claim = await this.prisma.claim.findUnique({
      where: { id },
      include: {
        equipment: true,
        client: {
          select: {
            id: true,
            firstName: true,
            lastName: true,
            company: true,
            email: true,
            phone: true,
            wilaya: true,
          },
        },
        files: {
          orderBy: { createdAt: 'asc' },
        },
        analysis: true,
        decision: {
          include: {
            investigator: {
              select: { firstName: true, lastName: true, email: true },
            },
          },
        },
      },
    });

    if (!claim) throw new NotFoundException('Claim not found');

    if (userRole === 'CLIENT' && claim.clientId !== userId) {
      throw new NotFoundException('Claim not found');
    }

    return claim;
  }

  /**
   * INVESTIGATOR: Submit APPROVED or REJECTED decision.
   * Fires the same /human-decision webhook as auto-decisions
   * so the same n8n workflow generates the PDF for all cases.
   */
  async submitDecision(
    claimId: string,
    investigatorId: string,
    dto: DecideClaimDto,
  ) {
    // Include equipment and analysis so webhook payload is complete for PDF
    const claim = await this.prisma.claim.findUnique({
      where: { id: claimId },
      include: {
        client: true,
        equipment: true,
        analysis: true,
      },
    });

    if (!claim) throw new NotFoundException('Claim not found');

    if (claim.status !== ClaimStatus.HUMAN_REVIEW) {
      throw new BadRequestException(
        `Cannot decide on a claim with status "${claim.status}". Claim must be in HUMAN_REVIEW.`,
      );
    }

    const newStatus =
      dto.outcome === DecisionOutcome.APPROVED
        ? ClaimStatus.APPROVED
        : ClaimStatus.REJECTED;

    await this.prisma.$transaction([
      this.prisma.decision.create({
        data: {
          claimId,
          type: DecisionType.HUMAN,
          outcome: dto.outcome,
          investigatorId,
          notes: dto.notes,
        },
      }),
      this.prisma.claim.update({
        where: { id: claimId },
        data: { status: newStatus },
      }),
    ]);

    this.logger.log(
      `Claim ${claim.reference} ${dto.outcome} by investigator ${investigatorId}`,
    );

    const outcomeText =
      dto.outcome === DecisionOutcome.APPROVED ? 'approuvé' : 'rejeté';

    await this.notifications.create(
      claim.clientId,
      `Sinistre ${outcomeText}`,
      `Votre sinistre ${claim.reference} a été ${outcomeText} après investigation.`,
    );

    /*
     * Fire to /human-decision — same endpoint used by auto-decisions.
     * Includes AI scores from the analysis record so the PDF shows
     * the full breakdown even for human decisions.
     */
    this.triggerWebhook('human-decision', {
      claimId,
      reference: claim.reference,
      clientId: claim.clientId,
      clientEmail: claim.client.email,
      clientName: `${claim.client.firstName} ${claim.client.lastName}`,
      clientPhone: claim.client.phone ?? 'N/A',
      clientWilaya: claim.client.wilaya ?? 'N/A',
      clientCompany: claim.client.company ?? 'N/A',
      equipmentName: claim.equipment?.name ?? 'N/A',
      equipmentType: claim.equipment?.type ?? 'N/A',
      incidentDate: claim.incidentDate,
      description: claim.description,
      claimedAmount: claim.claimedAmount,
      outcome: dto.outcome,
      notes: dto.notes,
      decidedAt: new Date().toISOString(),
      // AI scores from the analysis — present for human decisions too
      finalScore: claim.analysis?.finalScore ?? null,
      fraudClass: claim.analysis?.fraudClass ?? null,
      anomalyScore: claim.analysis?.anomalyScore ?? null,
      classificationScore: claim.analysis?.classificationScore ?? null,
      nlpScore: claim.analysis?.nlpScore ?? null,
      visionScore: claim.analysis?.visionScore ?? null,
    }).catch((err) =>
      this.logger.warn(`n8n webhook human-decision failed: ${err.message}`),
    );

    return {
      message: `Claim ${dto.outcome.toLowerCase()} successfully`,
      claimId,
      reference: claim.reference,
      newStatus,
    };
  }

  async savePdfUrl(claimId: string, pdfUrl: string) {
    await this.prisma.claim.update({
      where: { id: claimId },
      data: { pdfUrl },
    });

    this.logger.log(`PDF URL saved for claim ${claimId}`);

    return { message: 'PDF URL saved successfully', claimId, pdfUrl };
  }

  private detectFileType(file: Express.Multer.File): FileType {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
      return FileType.CSV;
    }
    if (file.mimetype.startsWith('image/')) {
      return FileType.PHOTO;
    }
    return FileType.PDF;
  }

  private async triggerWebhook(event: string, payload: object): Promise<void> {
    const base =
      this.config.get('N8N_WEBHOOK_BASE') || 'http://localhost:5678/webhook';
    await axios.post(`${base}/${event}`, payload, { timeout: 5000 });
  }
}