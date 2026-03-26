import { Test, TestingModule } from '@nestjs/testing';
import { NotificationsService } from './notifications.service';
import { PrismaService } from '../prisma/prisma.service';

/**
 * NotificationsService Tests
 * All Prisma calls are mocked — no real database connection needed.
 */
describe('NotificationsService', () => {
    let service: NotificationsService;
    let prisma: any;

    const mockNotification = {
        id: 'notif-123',
        userId: 'user-123',
        title: 'Sinistre reçu',
        message: 'Votre sinistre SIN-2026-ABC123 a été reçu.',
        isRead: false,
        createdAt: new Date(),
    };

    const mockReadNotification = {
        ...mockNotification,
        id: 'notif-456',
        isRead: true,
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                NotificationsService,
                {
                    provide: PrismaService,
                    useValue: {
                        notification: {
                            create: jest.fn(),
                            findMany: jest.fn(),
                            count: jest.fn(),
                            updateMany: jest.fn(),
                        },
                    },
                },
            ],
        }).compile();

        service = module.get<NotificationsService>(NotificationsService);
        prisma = module.get<PrismaService>(PrismaService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── create ───────────────────────────────────────────────────────────────

    describe('create', () => {
        it('should create a notification successfully', async () => {
            prisma.notification.create.mockResolvedValue(mockNotification);

            const result = await service.create(
                'user-123',
                'Sinistre reçu',
                'Votre sinistre SIN-2026-ABC123 a été reçu.',
            );

            expect(result).toEqual(mockNotification);
            expect(prisma.notification.create).toHaveBeenCalledWith({
                data: {
                    userId: 'user-123',
                    title: 'Sinistre reçu',
                    message: 'Votre sinistre SIN-2026-ABC123 a été reçu.',
                },
            });
        });
    });

    // ─── findAll ──────────────────────────────────────────────────────────────

    describe('findAll', () => {
        it('should return paginated notifications unread first', async () => {
            prisma.notification.findMany.mockResolvedValue([
                mockNotification,
                mockReadNotification,
            ]);
            prisma.notification.count.mockResolvedValue(2);

            const result = await service.findAll('user-123', { page: 1, limit: 20 });

            expect(result.data).toHaveLength(2);
            expect(result.pagination.total).toBe(2);
            expect(result.pagination.page).toBe(1);
        });

        it('should return empty list when user has no notifications', async () => {
            prisma.notification.findMany.mockResolvedValue([]);
            prisma.notification.count.mockResolvedValue(0);

            const result = await service.findAll('user-123', { page: 1, limit: 20 });

            expect(result.data).toHaveLength(0);
            expect(result.pagination.total).toBe(0);
        });

        it('should calculate pagination correctly', async () => {
            prisma.notification.findMany.mockResolvedValue([mockNotification]);
            prisma.notification.count.mockResolvedValue(45);

            const result = await service.findAll('user-123', { page: 3, limit: 20 });

            expect(result.pagination.totalPages).toBe(3);
            expect(result.pagination.hasNextPage).toBe(false);
            expect(result.pagination.hasPrevPage).toBe(true);
        });
    });

    // ─── countUnread ──────────────────────────────────────────────────────────

    describe('countUnread', () => {
        it('should return correct unread count', async () => {
            prisma.notification.count.mockResolvedValue(5);

            const result = await service.countUnread('user-123');

            expect(result).toEqual({ unread: 5 });
            expect(prisma.notification.count).toHaveBeenCalledWith({
                where: { userId: 'user-123', isRead: false },
            });
        });

        it('should return zero when all notifications are read', async () => {
            prisma.notification.count.mockResolvedValue(0);

            const result = await service.countUnread('user-123');

            expect(result).toEqual({ unread: 0 });
        });
    });

    // ─── markRead ─────────────────────────────────────────────────────────────

    describe('markRead', () => {
        it('should mark a notification as read', async () => {
            prisma.notification.updateMany.mockResolvedValue({ count: 1 });

            const result = await service.markRead('notif-123', 'user-123');

            expect(result).toEqual({ message: 'Notification marked as read' });
            expect(prisma.notification.updateMany).toHaveBeenCalledWith({
                where: { id: 'notif-123', userId: 'user-123' },
                data: { isRead: true },
            });
        });

        it('should not affect other users notifications', async () => {
            // updateMany with userId filter ensures only own notifications are updated
            prisma.notification.updateMany.mockResolvedValue({ count: 0 });

            const result = await service.markRead('notif-123', 'other-user-999');

            expect(result).toEqual({ message: 'Notification marked as read' });
            expect(prisma.notification.updateMany).toHaveBeenCalledWith({
                where: { id: 'notif-123', userId: 'other-user-999' },
                data: { isRead: true },
            });
        });
    });

    // ─── markAllRead ──────────────────────────────────────────────────────────

    describe('markAllRead', () => {
        it('should mark all unread notifications as read', async () => {
            prisma.notification.updateMany.mockResolvedValue({ count: 3 });

            const result = await service.markAllRead('user-123');

            expect(result).toEqual({
                message: 'All notifications marked as read',
                updated: 3,
            });
            expect(prisma.notification.updateMany).toHaveBeenCalledWith({
                where: { userId: 'user-123', isRead: false },
                data: { isRead: true },
            });
        });

        it('should return zero updated when all notifications already read', async () => {
            prisma.notification.updateMany.mockResolvedValue({ count: 0 });

            const result = await service.markAllRead('user-123');

            expect(result).toEqual({
                message: 'All notifications marked as read',
                updated: 0,
            });
        });
    });
});