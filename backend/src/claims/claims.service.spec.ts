import { Test, TestingModule } from '@nestjs/testing';
import { ClaimsService } from './claims.service';
import { PrismaService } from '../prisma/prisma.service';
import { MinioService } from '../files/minio.service';
import { QueueProducer } from '../queue/queue.producer';
import { NotificationsService } from '../notifications/notifications.service';
import { EquipmentService } from '../equipment/equipment.service';
import { ConfigService } from '@nestjs/config';
import {
    NotFoundException,
    ForbiddenException,
    BadRequestException,
} from '@nestjs/common';
import { ClaimStatus, DecisionOutcome } from '@prisma/client';

/**
 * ClaimsService Tests
 * All external dependencies are mocked.
 * No real database, MinIO, RabbitMQ, or n8n connections needed.
 */
describe('ClaimsService', () => {
    let service: ClaimsService;
    let prisma: any;
    let minio: any;
    let queueProducer: any;
    let notifications: any;
    let equipmentService: any;

    const mockEquipment = {
        id: 'equip-123',
        ownerId: 'user-123',
        name: 'Compresseur Atlas Copco GA-55',
        type: 'Compresseur',
        isActive: true,
    };

    const mockClaim = {
        id: 'claim-123',
        reference: 'SIN-2026-ABC123',
        clientId: 'user-123',
        equipmentId: 'equip-123',
        incidentDate: new Date('2026-01-15'),
        description: 'La pompe hydraulique a subi une surchauffe soudaine.',
        claimedAmount: 450000,
        status: ClaimStatus.PENDING,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockClaimInReview = {
        ...mockClaim,
        status: ClaimStatus.HUMAN_REVIEW,
        client: {
            id: 'user-123',
            firstName: 'Ahmed',
            lastName: 'Benali',
            email: 'ahmed@sonatrach.dz',
        },
    };

    const validDto = {
        equipmentId: 'equip-123',
        incidentDate: '2026-01-15',
        description: 'La pompe hydraulique a subi une surchauffe soudaine suite à un défaut.',
        claimedAmount: 450000,
    };

    const mockCsvFile: Partial<Express.Multer.File> = {
        originalname: 'sensors.csv',
        mimetype: 'text/csv',
        size: 1024,
        buffer: Buffer.from(''),
    };

    const mockPhotoFile: Partial<Express.Multer.File> = {
        originalname: 'damage.jpg',
        mimetype: 'image/jpeg',
        size: 2048,
        buffer: Buffer.from(''),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                ClaimsService,
                {
                    provide: PrismaService,
                    useValue: {
                        claim: {
                            create: jest.fn(),
                            findMany: jest.fn(),
                            findUnique: jest.fn(),
                            count: jest.fn(),
                            update: jest.fn(),
                        },
                        claimFile: {
                            create: jest.fn(),
                        },
                        decision: {
                            create: jest.fn(),
                        },
                        $transaction: jest.fn(),
                    },
                },
                {
                    provide: MinioService,
                    useValue: {
                        upload: jest.fn(),
                    },
                },
                {
                    provide: QueueProducer,
                    useValue: {
                        publishAnalysisJob: jest.fn(),
                    },
                },
                {
                    provide: NotificationsService,
                    useValue: {
                        create: jest.fn(),
                    },
                },
                {
                    provide: EquipmentService,
                    useValue: {
                        verifyActiveAndOwned: jest.fn(),
                    },
                },
                {
                    provide: ConfigService,
                    useValue: {
                        get: jest.fn().mockReturnValue('http://localhost:5678/webhook'),
                    },
                },
            ],
        }).compile();

        service = module.get<ClaimsService>(ClaimsService);
        prisma = module.get<PrismaService>(PrismaService);
        minio = module.get<MinioService>(MinioService);
        queueProducer = module.get<QueueProducer>(QueueProducer);
        notifications = module.get<NotificationsService>(NotificationsService);
        equipmentService = module.get<EquipmentService>(EquipmentService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── submitClaim ──────────────────────────────────────────────────────────

    describe('submitClaim', () => {
        it('should submit a claim successfully with CSV and photo', async () => {
            equipmentService.verifyActiveAndOwned.mockResolvedValue(mockEquipment);
            prisma.claim.create.mockResolvedValue(mockClaim);
            minio.upload.mockResolvedValue('claims/claim-123/sensors.csv');
            prisma.claimFile.create.mockResolvedValue({});
            queueProducer.publishAnalysisJob.mockResolvedValue(undefined);
            notifications.create.mockResolvedValue(undefined);

            const files = [mockCsvFile, mockPhotoFile] as Express.Multer.File[];
            const result = await service.submitClaim('user-123', validDto, files);

            expect(result.reference).toMatch(/^SIN-\d{4}-[A-Z0-9]{6}$/);
            expect(result.status).toBe(ClaimStatus.PENDING);
            expect(result.filesUploaded).toBe(2);
            expect(queueProducer.publishAnalysisJob).toHaveBeenCalledWith({
                claimId: 'claim-123',
            });
            expect(notifications.create).toHaveBeenCalledWith(
                'user-123',
                'Sinistre reçu',
                expect.stringContaining('SIN-'),
            );
        });

        it('should throw BadRequestException if incidentDate is in the future', async () => {
            const futureDate = new Date();
            futureDate.setFullYear(futureDate.getFullYear() + 1);

            const dto = {
                ...validDto,
                incidentDate: futureDate.toISOString().split('T')[0],
            };

            const files = [mockCsvFile, mockPhotoFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', dto, files)).rejects.toThrow(
                BadRequestException,
            );
            expect(equipmentService.verifyActiveAndOwned).not.toHaveBeenCalled();
        });

        it('should throw BadRequestException if no files are provided', async () => {
            equipmentService.verifyActiveAndOwned.mockResolvedValue(mockEquipment);

            await expect(service.submitClaim('user-123', validDto, [])).rejects.toThrow(
                BadRequestException,
            );
        });

        it('should throw BadRequestException if no CSV file is provided', async () => {
            equipmentService.verifyActiveAndOwned.mockResolvedValue(mockEquipment);

            const files = [mockPhotoFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', validDto, files)).rejects.toThrow(
                BadRequestException,
            );
        });

        it('should throw BadRequestException if no photo is provided', async () => {
            equipmentService.verifyActiveAndOwned.mockResolvedValue(mockEquipment);

            const files = [mockCsvFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', validDto, files)).rejects.toThrow(
                BadRequestException,
            );
        });

        it('should throw BadRequestException if a file exceeds 10MB', async () => {
            equipmentService.verifyActiveAndOwned.mockResolvedValue(mockEquipment);

            const largeFile: Partial<Express.Multer.File> = {
                originalname: 'huge.csv',
                mimetype: 'text/csv',
                size: 11 * 1024 * 1024, // 11MB
                buffer: Buffer.from(''),
            };

            const files = [largeFile, mockPhotoFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', validDto, files)).rejects.toThrow(
                BadRequestException,
            );
        });

        it('should propagate ForbiddenException if equipment does not belong to client', async () => {
            equipmentService.verifyActiveAndOwned.mockRejectedValue(
                new ForbiddenException('Not your equipment'),
            );

            const files = [mockCsvFile, mockPhotoFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', validDto, files)).rejects.toThrow(
                ForbiddenException,
            );
        });

        it('should propagate BadRequestException if equipment is deactivated', async () => {
            equipmentService.verifyActiveAndOwned.mockRejectedValue(
                new BadRequestException('Cannot submit a claim for deactivated equipment'),
            );

            const files = [mockCsvFile, mockPhotoFile] as Express.Multer.File[];
            await expect(service.submitClaim('user-123', validDto, files)).rejects.toThrow(
                BadRequestException,
            );
        });
    });

    // ─── findMyClaims ─────────────────────────────────────────────────────────

    describe('findMyClaims', () => {
        it('should return paginated claims for the client', async () => {
            prisma.claim.findMany.mockResolvedValue([mockClaim]);
            prisma.claim.count.mockResolvedValue(1);

            const result = await service.findMyClaims('user-123', { page: 1, limit: 10 });

            expect(result.data).toHaveLength(1);
            expect(result.pagination.total).toBe(1);
        });

        it('should return empty list when client has no claims', async () => {
            prisma.claim.findMany.mockResolvedValue([]);
            prisma.claim.count.mockResolvedValue(0);

            const result = await service.findMyClaims('user-123', { page: 1, limit: 10 });

            expect(result.data).toHaveLength(0);
            expect(result.pagination.total).toBe(0);
        });
    });

    // ─── getFlaggedClaims ─────────────────────────────────────────────────────

    describe('getFlaggedClaims', () => {
        it('should return paginated flagged claims for investigators', async () => {
            prisma.claim.findMany.mockResolvedValue([mockClaimInReview]);
            prisma.claim.count.mockResolvedValue(1);

            const result = await service.getFlaggedClaims({ page: 1, limit: 10 });

            expect(result.data).toHaveLength(1);
            expect(result.pagination.total).toBe(1);
        });
    });

    // ─── findOne ──────────────────────────────────────────────────────────────

    describe('findOne', () => {
        it('should return claim detail for the owner client', async () => {
            prisma.claim.findUnique.mockResolvedValue({
                ...mockClaim,
                clientId: 'user-123',
                equipment: mockEquipment,
                files: [],
                analysis: null,
                decision: null,
                client: { id: 'user-123', firstName: 'Ahmed', lastName: 'Benali' },
            });

            const result = await service.findOne('claim-123', 'user-123', 'CLIENT');

            expect(result.id).toBe('claim-123');
        });

        it('should return claim detail for any investigator', async () => {
            prisma.claim.findUnique.mockResolvedValue({
                ...mockClaim,
                clientId: 'user-123',
                equipment: mockEquipment,
                files: [],
                analysis: null,
                decision: null,
                client: { id: 'user-123', firstName: 'Ahmed', lastName: 'Benali' },
            });

            const result = await service.findOne('claim-123', 'inv-456', 'INVESTIGATOR');

            expect(result.id).toBe('claim-123');
        });

        it('should throw NotFoundException if client tries to view another users claim', async () => {
            prisma.claim.findUnique.mockResolvedValue({
                ...mockClaim,
                clientId: 'other-user-999',
            });

            await expect(
                service.findOne('claim-123', 'user-123', 'CLIENT'),
            ).rejects.toThrow(NotFoundException);
        });

        it('should throw NotFoundException if claim does not exist', async () => {
            prisma.claim.findUnique.mockResolvedValue(null);

            await expect(
                service.findOne('nonexistent-id', 'user-123', 'CLIENT'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    // ─── submitDecision ───────────────────────────────────────────────────────

    describe('submitDecision', () => {
        it('should approve a claim successfully', async () => {
            prisma.claim.findUnique.mockResolvedValue(mockClaimInReview);
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue(undefined);

            const result = await service.submitDecision('claim-123', 'inv-456', {
                outcome: DecisionOutcome.APPROVED,
                notes: 'Analyse confirme une vraie panne thermique.',
            });

            expect(result.newStatus).toBe(ClaimStatus.APPROVED);
            expect(result.message).toContain('approved');
            expect(notifications.create).toHaveBeenCalledWith(
                'user-123',
                'Sinistre approuvé',
                expect.stringContaining('SIN-2026-ABC123'),
            );
        });

        it('should reject a claim successfully', async () => {
            prisma.claim.findUnique.mockResolvedValue(mockClaimInReview);
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue(undefined);

            const result = await service.submitDecision('claim-123', 'inv-456', {
                outcome: DecisionOutcome.REJECTED,
                notes: 'Signes évidents de manipulation des données capteurs.',
            });

            expect(result.newStatus).toBe(ClaimStatus.REJECTED);
            expect(result.message).toContain('rejected');
        });

        it('should throw NotFoundException if claim does not exist', async () => {
            prisma.claim.findUnique.mockResolvedValue(null);

            await expect(
                service.submitDecision('nonexistent-id', 'inv-456', {
                    outcome: DecisionOutcome.APPROVED,
                    notes: 'Valid notes here.',
                }),
            ).rejects.toThrow(NotFoundException);
        });

        it('should throw BadRequestException if claim is not in HUMAN_REVIEW', async () => {
            prisma.claim.findUnique.mockResolvedValue({
                ...mockClaimInReview,
                status: ClaimStatus.PENDING,
            });

            await expect(
                service.submitDecision('claim-123', 'inv-456', {
                    outcome: DecisionOutcome.APPROVED,
                    notes: 'Valid notes here.',
                }),
            ).rejects.toThrow(BadRequestException);
        });

        it('should throw BadRequestException if claim is already approved', async () => {
            prisma.claim.findUnique.mockResolvedValue({
                ...mockClaimInReview,
                status: ClaimStatus.APPROVED,
            });

            await expect(
                service.submitDecision('claim-123', 'inv-456', {
                    outcome: DecisionOutcome.REJECTED,
                    notes: 'Trying to re-decide.',
                }),
            ).rejects.toThrow(BadRequestException);
        });
    });
});