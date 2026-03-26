import { Test, TestingModule } from '@nestjs/testing';
import { UsersService } from './users.service';
import { PrismaService } from '../prisma/prisma.service';
import { NotFoundException, ForbiddenException } from '@nestjs/common';

/**
 * UsersService Tests
 * All Prisma calls are mocked — no real database connection needed.
 */
describe('UsersService', () => {
    let service: UsersService;
    let prisma: any;

    // reusable mock user object
    const mockUser = {
        id: 'user-123',
        email: 'ahmed@sonatrach.dz',
        firstName: 'Ahmed',
        lastName: 'Benali',
        role: 'CLIENT',
        phone: '+213555123456',
        wilaya: 'Boumerdès',
        commune: 'Boumerdès centre',
        company: 'Sonatrach',
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const mockInvestigator = {
        id: 'inv-456',
        email: 'karim@caat.dz',
        firstName: 'Karim',
        lastName: 'Meziani',
        role: 'INVESTIGATOR',
        phone: '+213770000001',
        wilaya: 'Alger',
        commune: null,
        company: 'CAAT Insurance',
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                UsersService,
                {
                    provide: PrismaService,
                    useValue: {
                        user: {
                            findUnique: jest.fn(),
                            update: jest.fn(),
                        },
                        claim: {
                            count: jest.fn(),
                        },
                    },
                },
            ],
        }).compile();

        service = module.get<UsersService>(UsersService);
        prisma = module.get<PrismaService>(PrismaService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── getMe ────────────────────────────────────────────────────────────────

    describe('getMe', () => {
        it('should return the authenticated user profile', async () => {
            prisma.user.findUnique.mockResolvedValue(mockUser);

            const result = await service.getMe('user-123');

            expect(result).toEqual(mockUser);
            expect(prisma.user.findUnique).toHaveBeenCalledWith({
                where: { id: 'user-123' },
                select: expect.objectContaining({ email: true, firstName: true }),
            });
        });

        it('should throw NotFoundException if user does not exist', async () => {
            prisma.user.findUnique.mockResolvedValue(null);

            await expect(service.getMe('nonexistent-id')).rejects.toThrow(NotFoundException);
        });
    });

    // ─── updateMe ─────────────────────────────────────────────────────────────

    describe('updateMe', () => {
        it('should update and return the user profile', async () => {
            const dto = { firstName: 'Mohamed', wilaya: 'Alger' };
            const updatedUser = { ...mockUser, ...dto };

            prisma.user.findUnique.mockResolvedValue(mockUser);
            prisma.user.update.mockResolvedValue(updatedUser);

            const result = await service.updateMe('user-123', dto);

            expect(result).toEqual(updatedUser);
            expect(prisma.user.update).toHaveBeenCalledWith({
                where: { id: 'user-123' },
                data: dto,
                select: expect.objectContaining({ email: true }),
            });
        });

        it('should throw NotFoundException if user does not exist', async () => {
            prisma.user.findUnique.mockResolvedValue(null);

            await expect(service.updateMe('nonexistent-id', {})).rejects.toThrow(NotFoundException);
        });
    });

    // ─── getUserById ──────────────────────────────────────────────────────────

    describe('getUserById', () => {
        it('should return a client profile when requested by an investigator', async () => {
            prisma.user.findUnique.mockResolvedValue({
                ...mockUser,
                _count: { claims: 3 },
            });

            const result = await service.getUserById('inv-456', 'INVESTIGATOR', 'user-123');

            expect(result).toHaveProperty('email', 'ahmed@sonatrach.dz');
            expect(result).toHaveProperty('_count');
        });

        it('should throw ForbiddenException if requester is not an investigator', async () => {
            await expect(
                service.getUserById('user-123', 'CLIENT', 'user-456'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should throw NotFoundException if target user does not exist', async () => {
            prisma.user.findUnique.mockResolvedValue(null);

            await expect(
                service.getUserById('inv-456', 'INVESTIGATOR', 'nonexistent-id'),
            ).rejects.toThrow(NotFoundException);
        });
    });

    // ─── getMyStats ───────────────────────────────────────────────────────────

    describe('getMyStats', () => {
        it('should return correct claim statistics for a client', async () => {
            // mock counts in order: total, approved, rejected, pending, analyzing, humanReview
            prisma.claim.count
                .mockResolvedValueOnce(10)
                .mockResolvedValueOnce(4)
                .mockResolvedValueOnce(3)
                .mockResolvedValueOnce(2)
                .mockResolvedValueOnce(1)
                .mockResolvedValueOnce(0);

            const result = await service.getMyStats('user-123');

            expect(result).toEqual({
                total: 10,
                approved: 4,
                rejected: 3,
                pending: 2,
                analyzing: 1,
                humanReview: 0,
                approvalRate: 40,
            });
        });

        it('should return 0 approval rate when client has no claims', async () => {
            prisma.claim.count.mockResolvedValue(0);

            const result = await service.getMyStats('user-123');

            expect(result.total).toBe(0);
            expect(result.approvalRate).toBe(0);
        });
    });
});