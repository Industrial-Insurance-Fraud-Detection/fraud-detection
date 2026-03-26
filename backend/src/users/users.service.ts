import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { UpdateUserDto } from './dto/update-user.dto';

/**
 * UsersService
 * Handles profile management for clients and investigators.
 * Investigators can view any client profile when reviewing a claim.
 * Clients can only view and update their own profile.
 */
@Injectable()
export class UsersService {
  constructor(private readonly prisma: PrismaService) { }

  /**
   * Returns the full profile of the currently authenticated user.
   * Excludes password hash from the response.
   */
  async getMe(userId: string) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        role: true,
        phone: true,
        wilaya: true,
        commune: true,
        company: true,
        createdAt: true,
        updatedAt: true,
      },
    });

    if (!user) throw new NotFoundException('User not found');
    return user;
  }

  /**
   * Updates the profile of the currently authenticated user.
   * Role and email cannot be changed via this endpoint.
   */
  async updateMe(userId: string, dto: UpdateUserDto) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) throw new NotFoundException('User not found');

    return this.prisma.user.update({
      where: { id: userId },
      data: dto,
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        role: true,
        phone: true,
        wilaya: true,
        commune: true,
        company: true,
        updatedAt: true,
      },
    });
  }

  /**
   * Returns a client's public profile.
   * Only accessible by investigators when reviewing a claim.
   * Sensitive fields like password and audit logs are excluded.
   */
  async getUserById(requesterId: string, requesterRole: string, targetUserId: string) {
    // only investigators can view other users profiles
    if (requesterRole !== 'INVESTIGATOR') {
      throw new ForbiddenException('Access denied');
    }

    const user = await this.prisma.user.findUnique({
      where: { id: targetUserId },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        role: true,
        phone: true,
        wilaya: true,
        commune: true,
        company: true,
        createdAt: true,
        _count: {
          select: { claims: true },
        },
      },
    });

    if (!user) throw new NotFoundException('User not found');
    return user;
  }

  /**
   * Returns a summary of the authenticated client's claim statistics.
   * Used by the client dashboard to show claim history at a glance.
   */
  async getMyStats(userId: string) {
    const [total, approved, rejected, pending, analyzing, humanReview] =
      await Promise.all([
        this.prisma.claim.count({ where: { clientId: userId } }),
        this.prisma.claim.count({ where: { clientId: userId, status: 'APPROVED' } }),
        this.prisma.claim.count({ where: { clientId: userId, status: 'REJECTED' } }),
        this.prisma.claim.count({ where: { clientId: userId, status: 'PENDING' } }),
        this.prisma.claim.count({ where: { clientId: userId, status: 'ANALYZING' } }),
        this.prisma.claim.count({ where: { clientId: userId, status: 'HUMAN_REVIEW' } }),
      ]);

    return {
      total,
      approved,
      rejected,
      pending,
      analyzing,
      humanReview,
      approvalRate: total > 0 ? Math.round((approved / total) * 100) : 0,
    };
  }
}