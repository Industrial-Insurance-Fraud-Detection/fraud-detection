import {
  Controller,
  Get,
  Post,
  Patch,
  Param,
  Body,
  Query,
  UseGuards,
} from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiBearerAuth,
  ApiQuery,
  ApiResponse,
} from '@nestjs/swagger';
import { NotificationsService } from './notifications.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { PaginationDto } from '../common/dto/pagination.dto';
import { Public } from '../common/decorators/public.decorator';

/**
 * NotificationsController
 * All endpoints require authentication except sendInternal.
 * No role restriction — both clients and investigators receive notifications.
 */
@ApiTags('Notifications')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('notifications')
export class NotificationsController {
  constructor(private readonly notificationsService: NotificationsService) { }

  /**
   * GET /notifications
   * Returns paginated notifications for the authenticated user.
   * Unread notifications appear first.
   */
  @Get()
  @ApiOperation({ summary: 'List my notifications with pagination' })
  @ApiQuery({ name: 'page', required: false, type: Number })
  @ApiQuery({ name: 'limit', required: false, type: Number })
  @ApiResponse({ status: 200, description: 'Paginated notifications returned' })
  findAll(
    @CurrentUser() user: any,
    @Query() pagination: PaginationDto,
  ) {
    return this.notificationsService.findAll(user.id, pagination);
  }

  /**
   * GET /notifications/unread-count
   * Returns the count of unread notifications.
   * Used by frontend to show badge count on notification icon.
   */
  @Get('unread-count')
  @ApiOperation({ summary: 'Get count of unread notifications' })
  @ApiResponse({ status: 200, description: 'Unread count returned' })
  countUnread(@CurrentUser() user: any) {
    return this.notificationsService.countUnread(user.id);
  }

  /**
   * POST /notifications/internal
   * Called by n8n to send a notification to a specific user.
   * Used to notify clients when their decision letter PDF is ready.
   * No auth required — internal n8n service call.
   */
  @Post('internal')
  @Public()
  @ApiOperation({ summary: 'Send notification to a user (called by n8n — no auth required)' })
  @ApiResponse({ status: 201, description: 'Notification sent successfully' })
  async sendInternal(
    @Body() body: { userId: string; title: string; message: string },
  ) {
    return this.notificationsService.create(
      body.userId,
      body.title,
      body.message,
    );
  }

  /**
   * PATCH /notifications/:id/read
   * Marks a single notification as read.
   * Only marks notifications belonging to the authenticated user.
   */
  @Patch(':id/read')
  @ApiOperation({ summary: 'Mark a notification as read' })
  @ApiResponse({ status: 200, description: 'Notification marked as read' })
  markRead(@Param('id') id: string, @CurrentUser() user: any) {
    return this.notificationsService.markRead(id, user.id);
  }

  /**
   * PATCH /notifications/read-all
   * Marks all notifications as read for the authenticated user.
   */
  @Patch('read-all')
  @ApiOperation({ summary: 'Mark all notifications as read' })
  @ApiResponse({ status: 200, description: 'All notifications marked as read' })
  markAllRead(@CurrentUser() user: any) {
    return this.notificationsService.markAllRead(user.id);
  }
}