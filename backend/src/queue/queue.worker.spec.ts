import { Test, TestingModule } from '@nestjs/testing';
import { QueueWorker } from './queue.worker';
import { PrismaService } from '../prisma/prisma.service';
import { NotificationsService } from '../notifications/notifications.service';
import { ConfigService } from '@nestjs/config';
import { ClaimStatus, DecisionOutcome } from '@prisma/client';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

/**
 * QueueWorker Tests
 * Tests the AI orchestration logic — scoring, routing, auto-decision.
 * Axios calls to AI services are mocked.
 */
describe('QueueWorker', () => {
    let worker: QueueWorker;
    let prisma: any;
    let notifications: any;
    let config: any;

    const mockClaim = {
        id: 'claim-123',
        reference: 'SIN-2026-ABC123',
        clientId: 'user-123',
        equipmentId: 'equip-123',
        description: 'La pompe hydraulique a subi une surchauffe.',
        claimedAmount: 450000,
        incidentDate: new Date('2026-01-15'),
        status: ClaimStatus.PENDING,
        equipment: {
            id: 'equip-123',
            type: 'Compresseur',
            name: 'Atlas Copco GA-55',
        },
        files: [
            {
                id: 'file-1',
                fileType: 'CSV',
                minioPath: 'claims/claim-123/sensors.csv',
            },
            {
                id: 'file-2',
                fileType: 'PHOTO',
                minioPath: 'claims/claim-123/damage.jpg',
            },
        ],
        client: {
            id: 'user-123',
            email: 'ahmed@sonatrach.dz',
            firstName: 'Ahmed',
            lastName: 'Benali',
        },
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                QueueWorker,
                {
                    provide: PrismaService,
                    useValue: {
                        claim: {
                            update: jest.fn(),
                            findUnique: jest.fn(),
                        },
                        aIAnalysis: {
                            create: jest.fn(),
                        },
                        decision: {
                            create: jest.fn(),
                        },
                        $transaction: jest.fn(),
                    },
                },
                {
                    provide: NotificationsService,
                    useValue: {
                        create: jest.fn(),
                    },
                },
                {
                    provide: ConfigService,
                    useValue: {
                        get: jest.fn().mockImplementation((key: string) => {
                            const map: Record<string, string> = {
                                AI_ANOMALY_URL: 'http://localhost:8001',
                                AI_CLASSIFICATION_URL: 'http://localhost:8002',
                                AI_NLP_URL: 'http://localhost:8003',
                                AI_VISION_URL: 'http://localhost:8004',
                                N8N_WEBHOOK_BASE: 'http://localhost:5678/webhook',
                            };
                            return map[key];
                        }),
                    },
                },
            ],
        }).compile();

        worker = module.get<QueueWorker>(QueueWorker);
        prisma = module.get<PrismaService>(PrismaService);
        notifications = module.get<NotificationsService>(NotificationsService);
        config = module.get<ConfigService>(ConfigService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── score extraction ─────────────────────────────────────────────────────

    describe('extractScore (private — tested via handleAnalysisJob)', () => {
        it('should use neutral score 50 when all AI services fail', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(mockClaim);
            prisma.aIAnalysis.create.mockResolvedValue({});
            prisma.claim.update.mockResolvedValue({});
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue({});

            // all AI services fail
            mockedAxios.post.mockRejectedValue(new Error('Service unavailable'));

            await worker.handleAnalysisJob({ claimId: 'claim-123' });

            // all 4 services failed → all scores = 50 → finalScore = 50 → HUMAN_REVIEW
            expect(prisma.aIAnalysis.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    data: expect.objectContaining({
                        anomalyScore: 50,
                        classificationScore: 50,
                        nlpScore: 50,
                        visionScore: 50,
                        finalScore: 50,
                        fraudClass: 'MEDIUM',
                    }),
                }),
            );
        });
    });

    // ─── auto-approve ─────────────────────────────────────────────────────────

    describe('handleAnalysisJob — auto-approve (score < 30)', () => {
        it('should auto-approve claim when all AI services return low fraud scores', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(mockClaim);
            prisma.aIAnalysis.create.mockResolvedValue({});
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue({});

            // all services return low scores → weighted average = 20
            mockedAxios.post.mockResolvedValue({ data: { score: 20 } });

            await worker.handleAnalysisJob({ claimId: 'claim-123' });

            expect(prisma.aIAnalysis.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    data: expect.objectContaining({
                        finalScore: 20,
                        fraudClass: 'LOW',
                    }),
                }),
            );

            // transaction should create AUTO APPROVED decision
            expect(prisma.$transaction).toHaveBeenCalled();

            // client should be notified of approval
            expect(notifications.create).toHaveBeenCalledWith(
                'user-123',
                'Sinistre approuvé automatiquement',
                expect.stringContaining('approuvé'),
            );
        });
    });

    // ─── auto-reject ──────────────────────────────────────────────────────────

    describe('handleAnalysisJob — auto-reject (score >= 70)', () => {
        it('should auto-reject claim when all AI services return high fraud scores', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(mockClaim);
            prisma.aIAnalysis.create.mockResolvedValue({});
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue({});

            // all services return high scores → weighted average = 85
            mockedAxios.post.mockResolvedValue({ data: { score: 85 } });

            await worker.handleAnalysisJob({ claimId: 'claim-123' });

            expect(prisma.aIAnalysis.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    data: expect.objectContaining({
                        finalScore: 85,
                        fraudClass: 'HIGH',
                    }),
                }),
            );

            expect(notifications.create).toHaveBeenCalledWith(
                'user-123',
                'Sinistre rejeté automatiquement',
                expect.stringContaining('rejeté'),
            );
        });
    });

    // ─── human review ─────────────────────────────────────────────────────────

    describe('handleAnalysisJob — human review (score 30–69)', () => {
        it('should route to HUMAN_REVIEW when score is in medium range', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(mockClaim);
            prisma.aIAnalysis.create.mockResolvedValue({});
            notifications.create.mockResolvedValue({});

            // all services return medium scores → weighted average = 50
            mockedAxios.post.mockResolvedValue({ data: { score: 50 } });

            await worker.handleAnalysisJob({ claimId: 'claim-123' });

            // claim status should be updated to HUMAN_REVIEW
            expect(prisma.claim.update).toHaveBeenCalledWith(
                expect.objectContaining({
                    data: { status: ClaimStatus.HUMAN_REVIEW },
                }),
            );

            // no auto decision transaction
            expect(prisma.$transaction).not.toHaveBeenCalled();

            // client notified about human review
            expect(notifications.create).toHaveBeenCalledWith(
                'user-123',
                "Sinistre en cours d'examen",
                expect.stringContaining('humaine'),
            );
        });
    });

    // ─── partial service failure ──────────────────────────────────────────────

    describe('handleAnalysisJob — partial AI service failure', () => {
        it('should continue processing when some AI services fail', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(mockClaim);
            prisma.aIAnalysis.create.mockResolvedValue({});
            prisma.$transaction.mockResolvedValue([{}, {}]);
            notifications.create.mockResolvedValue({});

            // anomaly and vision succeed with score 20
            // classification and nlp fail → default to 50
            mockedAxios.post
                .mockResolvedValueOnce({ data: { score: 20 } }) // anomaly
                .mockRejectedValueOnce(new Error('timeout'))     // classification
                .mockRejectedValueOnce(new Error('timeout'))     // nlp
                .mockResolvedValueOnce({ data: { score: 20 } }); // vision

            await worker.handleAnalysisJob({ claimId: 'claim-123' });

            // finalScore = 20*0.35 + 50*0.25 + 50*0.20 + 20*0.20 = 7 + 12.5 + 10 + 4 = 33.5
            // fraudClass = MEDIUM → HUMAN_REVIEW
            expect(prisma.aIAnalysis.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    data: expect.objectContaining({
                        anomalyScore: 20,
                        classificationScore: 50,
                        nlpScore: 50,
                        visionScore: 20,
                        fraudClass: 'MEDIUM',
                    }),
                }),
            );
        });
    });

    // ─── claim not found ──────────────────────────────────────────────────────

    describe('handleAnalysisJob — claim not found', () => {
        it('should return Nack without processing if claim does not exist', async () => {
            prisma.claim.update.mockResolvedValue({});
            prisma.claim.findUnique.mockResolvedValue(null);

            const result = await worker.handleAnalysisJob({ claimId: 'nonexistent-id' });

            // should return Nack to discard the message
            expect(result).toBeDefined();
            expect(prisma.aIAnalysis.create).not.toHaveBeenCalled();
        });
    });
});