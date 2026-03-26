import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { PaginationDto, paginate } from '../common/dto/pagination.dto';

/**
 * NotificationsService
 * Handles in-app notifications for both clients and investigators.
 * Notifications are created internally by ClaimsService at every status change.
 * Users can list, count unread, mark one as read, or mark all as read.
 */
@Injectable()
export class NotificationsService {
  private readonly logger = new Logger(NotificationsService.name);

  constructor(private readonly prisma: PrismaService) { }

  /**
   * Creates a notification for a user.
   * Called internally by ClaimsService — not exposed as an HTTP endpoint.
   */
  async create(userId: string, title: string, message: string) {
    const notification = await this.prisma.notification.create({
      data: { userId, title, message },
    });
    this.logger.log(`Notification created for user ${userId}: ${title}`);
    return notification;
  }

  /**
   * Returns paginated notifications for the authenticated user.
   * Unread notifications appear first, then sorted by most recent.
   */
  async findAll(userId: string, pagination: PaginationDto) {
    const { page = 1, limit = 20 } = pagination;
    const skip = (page - 1) * limit;

    const [data, total] = await Promise.all([
      this.prisma.notification.findMany({
        where: { userId },
        skip,
        take: limit,
        orderBy: [{ isRead: 'asc' }, { createdAt: 'desc' }],
      }),
      this.prisma.notification.count({ where: { userId } }),
    ]);

    return paginate(data, total, page, limit);
  }

  /**
   * Returns the count of unread notifications for the authenticated user.
   * Used by the frontend to display a badge on the notifications icon.
   */
  async countUnread(userId: string) {
    const count = await this.prisma.notification.count({
      where: { userId, isRead: false },
    });
    return { unread: count };
  }

  /**
   * Marks a single notification as read.
   * Uses updateMany with userId filter to prevent marking other users notifications.
   */
  async markRead(id: string, userId: string) {
    await this.prisma.notification.updateMany({
      where: { id, userId },
      data: { isRead: true },
    });
    return { message: 'Notification marked as read' };
  }

  /**
   * Marks all unread notifications as read for the authenticated user.
   */
  async markAllRead(userId: string) {
    const { count } = await this.prisma.notification.updateMany({
      where: { userId, isRead: false },
      data: { isRead: true },
    });
    return { message: 'All notifications marked as read', updated: count };
  }
}