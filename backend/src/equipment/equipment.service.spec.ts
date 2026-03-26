import { Test, TestingModule } from '@nestjs/testing';
import { EquipmentService } from './equipment.service';
import { PrismaService } from '../prisma/prisma.service';
import {
    NotFoundException,
    ForbiddenException,
    ConflictException,
    BadRequestException,
} from '@nestjs/common';

/**
 * EquipmentService Tests
 * All Prisma calls are mocked — no real database connection needed.
 */
describe('EquipmentService', () => {
    let service: EquipmentService;
    let prisma: any;

    // reusable mock equipment object
    const mockEquipment = {
        id: 'equip-123',
        ownerId: 'user-123',
        name: 'Compresseur Atlas Copco GA-55',
        type: 'Compresseur',
        manufacturer: 'Atlas Copco',
        model: 'GA-55',
        serialNumber: 'AC-GA55-2019-001',
        commissionDate: new Date('2019-06-15'),
        location: 'Usine Boumerdès — Bâtiment B2',
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
    };

    const validDto = {
        name: 'Compresseur Atlas Copco GA-55',
        type: 'Compresseur',
        manufacturer: 'Atlas Copco',
        model: 'GA-55',
        serialNumber: 'AC-GA55-2019-001',
        commissionDate: '2019-06-15',
        location: 'Usine Boumerdès — Bâtiment B2',
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                EquipmentService,
                {
                    provide: PrismaService,
                    useValue: {
                        equipment: {
                            findUnique: jest.fn(),
                            findMany: jest.fn(),
                            count: jest.fn(),
                            create: jest.fn(),
                            update: jest.fn(),
                        },
                    },
                },
            ],
        }).compile();

        service = module.get<EquipmentService>(EquipmentService);
        prisma = module.get<PrismaService>(PrismaService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── create ───────────────────────────────────────────────────────────────

    describe('create', () => {
        it('should register a new machine successfully', async () => {
            prisma.equipment.findUnique.mockResolvedValue(null);
            prisma.equipment.create.mockResolvedValue(mockEquipment);

            const result = await service.create('user-123', validDto);

            expect(result).toEqual(mockEquipment);
            expect(prisma.equipment.create).toHaveBeenCalledTimes(1);
        });

        it('should throw ConflictException if serial number already exists', async () => {
            prisma.equipment.findUnique.mockResolvedValue(mockEquipment);

            await expect(service.create('user-123', validDto)).rejects.toThrow(ConflictException);
            expect(prisma.equipment.create).not.toHaveBeenCalled();
        });

        it('should throw BadRequestException if commissionDate is in the future', async () => {
            const futureDate = new Date();
            futureDate.setFullYear(futureDate.getFullYear() + 1);

            const dto = {
                ...validDto,
                commissionDate: futureDate.toISOString().split('T')[0],
            };

            await expect(service.create('user-123', dto)).rejects.toThrow(BadRequestException);
            expect(prisma.equipment.findUnique).not.toHaveBeenCalled();
        });
    });

    // ─── findAllForOwner ──────────────────────────────────────────────────────

    describe('findAllForOwner', () => {
        it('should return paginated equipment list', async () => {
            prisma.equipment.findMany.mockResolvedValue([mockEquipment]);
            prisma.equipment.count.mockResolvedValue(1);

            const result = await service.findAllForOwner('user-123', { page: 1, limit: 10 });

            expect(result.data).toHaveLength(1);
            expect(result.pagination.total).toBe(1);
            expect(result.pagination.page).toBe(1);
            expect(result.pagination.totalPages).toBe(1);
        });

        it('should return empty list when owner has no equipment', async () => {
            prisma.equipment.findMany.mockResolvedValue([]);
            prisma.equipment.count.mockResolvedValue(0);

            const result = await service.findAllForOwner('user-123', { page: 1, limit: 10 });

            expect(result.data).toHaveLength(0);
            expect(result.pagination.total).toBe(0);
        });

        it('should calculate pagination metadata correctly', async () => {
            prisma.equipment.findMany.mockResolvedValue([mockEquipment]);
            prisma.equipment.count.mockResolvedValue(25);

            const result = await service.findAllForOwner('user-123', { page: 2, limit: 10 });

            expect(result.pagination.totalPages).toBe(3);
            expect(result.pagination.hasNextPage).toBe(true);
            expect(result.pagination.hasPrevPage).toBe(true);
        });
    });

    // ─── findOne ──────────────────────────────────────────────────────────────

    describe('findOne', () => {
        it('should return equipment when it belongs to the owner', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                claims: [],
                _count: { claims: 0 },
            });

            const result = await service.findOne('equip-123', 'user-123');

            expect(result.id).toBe('equip-123');
        });

        it('should throw NotFoundException if equipment does not exist', async () => {
            prisma.equipment.findUnique.mockResolvedValue(null);

            await expect(service.findOne('nonexistent-id', 'user-123')).rejects.toThrow(
                NotFoundException,
            );
        });

        it('should throw ForbiddenException if equipment belongs to another user', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                ownerId: 'other-user-999',
            });

            await expect(service.findOne('equip-123', 'user-123')).rejects.toThrow(
                ForbiddenException,
            );
        });
    });

    // ─── update ───────────────────────────────────────────────────────────────

    describe('update', () => {
        it('should update equipment successfully', async () => {
            const updated = { ...mockEquipment, location: 'Bâtiment C1' };

            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                claims: [],
                _count: { claims: 0 },
            });
            prisma.equipment.update.mockResolvedValue(updated);

            const result = await service.update('equip-123', 'user-123', {
                location: 'Bâtiment C1',
            });

            expect(result.location).toBe('Bâtiment C1');
        });

        it('should throw BadRequestException if updated commissionDate is in the future', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                claims: [],
                _count: { claims: 0 },
            });

            const futureDate = new Date();
            futureDate.setFullYear(futureDate.getFullYear() + 1);

            await expect(
                service.update('equip-123', 'user-123', {
                    commissionDate: futureDate.toISOString().split('T')[0],
                }),
            ).rejects.toThrow(BadRequestException);
        });
    });

    // ─── remove ───────────────────────────────────────────────────────────────

    describe('remove', () => {
        it('should soft delete equipment by setting isActive to false', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                claims: [],
                _count: { claims: 0 },
            });
            prisma.equipment.update.mockResolvedValue({
                ...mockEquipment,
                isActive: false,
            });

            const result = await service.remove('equip-123', 'user-123');

            expect(result.isActive).toBe(false);
            expect(prisma.equipment.update).toHaveBeenCalledWith({
                where: { id: 'equip-123' },
                data: { isActive: false },
            });
        });
    });

    // ─── verifyActiveAndOwned ─────────────────────────────────────────────────

    describe('verifyActiveAndOwned', () => {
        it('should return equipment when active and owned by the client', async () => {
            prisma.equipment.findUnique.mockResolvedValue(mockEquipment);

            const result = await service.verifyActiveAndOwned('equip-123', 'user-123');

            expect(result.id).toBe('equip-123');
        });

        it('should throw NotFoundException if equipment does not exist', async () => {
            prisma.equipment.findUnique.mockResolvedValue(null);

            await expect(
                service.verifyActiveAndOwned('nonexistent-id', 'user-123'),
            ).rejects.toThrow(NotFoundException);
        });

        it('should throw ForbiddenException if equipment belongs to another user', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                ownerId: 'other-user-999',
            });

            await expect(
                service.verifyActiveAndOwned('equip-123', 'user-123'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should throw BadRequestException if equipment is deactivated', async () => {
            prisma.equipment.findUnique.mockResolvedValue({
                ...mockEquipment,
                isActive: false,
            });

            await expect(
                service.verifyActiveAndOwned('equip-123', 'user-123'),
            ).rejects.toThrow(BadRequestException);
        });
    });
});