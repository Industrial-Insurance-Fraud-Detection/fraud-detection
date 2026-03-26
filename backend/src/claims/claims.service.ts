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

// file upload limits
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB per file
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

  /**
   * CLIENT: Submit a new insurance claim with supporting files.
   * Validates equipment ownership and active status.
   * Requires at least 1 CSV sensor file and 1 photo.
   * Files are uploaded to MinIO, job is pushed to RabbitMQ.
   */
  async submitClaim(
    clientId: string,
    dto: CreateClaimDto,
    files: Express.Multer.File[],
  ) {
    // validate incident date is not in the future
    const incidentDate = new Date(dto.incidentDate);
    if (incidentDate > new Date()) {
      throw new BadRequestException('incidentDate cannot be in the future');
    }

    // validate equipment exists, is active, and belongs to this client
    await this.equipmentService.verifyActiveAndOwned(dto.equipmentId, clientId);

    // validate file count
    if (!files || files.length === 0) {
      throw new BadRequestException('At least one file is required');
    }
    if (files.length > MAX_FILES) {
      throw new BadRequestException(`Cannot upload more than ${MAX_FILES} files`);
    }

    // categorize and validate files
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

    // validate individual file sizes
    for (const file of files) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        throw new BadRequestException(
          `File "${file.originalname}" exceeds the 10MB size limit`,
        );
      }
    }

    // generate unique claim reference
    const reference = `SIN-${new Date().getFullYear()}-${uuidv4().slice(0, 6).toUpperCase()}`;

    // save claim record to database
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

    // upload all files to MinIO and record each in the database
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

    // push async AI analysis job to RabbitMQ — non-blocking
    await this.queueProducer.publishAnalysisJob({ claimId: claim.id });
    this.logger.log(`Analysis job queued for claim ${reference}`);

    // notify client that claim was received
    await this.notifications.create(
      clientId,
      'Sinistre reçu',
      `Votre sinistre ${reference} a été reçu et est en cours d'analyse IA.`,
    );

    // trigger n8n webhook — non-blocking, failure does not affect response
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

  /**
   * CLIENT: Returns a paginated list of the authenticated client's claims.
   * Sorted by most recent first.
   */
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

  /**
   * INVESTIGATOR: Returns paginated claims awaiting human review.
   * Sorted by highest fraud score first — most urgent cases at the top.
   */
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

  /**
   * BOTH: Returns full claim detail including files, AI analysis, and decision.
   * CLIENT can only view their own claims.
   * INVESTIGATOR can view any claim.
   */
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

    // clients can only see their own claims
    // return 404 instead of 403 to avoid revealing claim existence
    if (userRole === 'CLIENT' && claim.clientId !== userId) {
      throw new NotFoundException('Claim not found');
    }

    return claim;
  }

  /**
   * INVESTIGATOR: Submit APPROVED or REJECTED decision with mandatory notes.
   * Claim must be in HUMAN_REVIEW status.
   * Decision and status update are atomic — both succeed or both fail.
   */
  async submitDecision(
    claimId: string,
    investigatorId: string,
    dto: DecideClaimDto,
  ) {
    const claim = await this.prisma.claim.findUnique({
      where: { id: claimId },
      include: { client: true },
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

    // atomic transaction — decision record + claim status update
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

    // notify client of the decision
    const outcomeText =
      dto.outcome === DecisionOutcome.APPROVED ? 'approuvé' : 'rejeté';
    await this.notifications.create(
      claim.clientId,
      `Sinistre ${outcomeText}`,
      `Votre sinistre ${claim.reference} a été ${outcomeText} après investigation.`,
    );

    // trigger n8n webhook — non-blocking
    this.triggerWebhook('human-decision', {
      claimId,
      reference: claim.reference,
      decision: dto.outcome,
      notes: dto.notes,
      clientEmail: claim.client.email,
      clientName: `${claim.client.firstName} ${claim.client.lastName}`,
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

  /**
   * Stores the decision letter PDF URL on the claim record.
   * Called by n8n after generating and saving the PDF to MinIO.
   */
  async savePdfUrl(claimId: string, pdfUrl: string) {
    await this.prisma.claim.update({
      where: { id: claimId },
      data: { pdfUrl },
    });

    this.logger.log(`PDF URL saved for claim ${claimId}`);

    return { message: 'PDF URL saved successfully', claimId, pdfUrl };
  }

  /**
   * Detects the file type from MIME type or file extension.
   * Used when saving file records to the database.
   */
  private detectFileType(file: Express.Multer.File): FileType {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
      return FileType.CSV;
    }
    if (file.mimetype.startsWith('image/')) {
      return FileType.PHOTO;
    }
    return FileType.PDF;
  }

  /**
   * Triggers an n8n automation webhook for workflow integration.
   * Non-blocking — called with .catch() so failures never affect the response.
   */
  private async triggerWebhook(event: string, payload: object): Promise<void> {
    const base =
      this.config.get('N8N_WEBHOOK_BASE') || 'http://localhost:5678/webhook';
    await axios.post(`${base}/${event}`, payload, { timeout: 5000 });
  }
}