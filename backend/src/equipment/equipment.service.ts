import {
  Injectable,
  NotFoundException,
  ForbiddenException,
  ConflictException,
  BadRequestException,
} from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { CreateEquipmentDto } from './dto/create-equipment.dto';
import { UpdateEquipmentDto } from './dto/update-equipment.dto';
import { PaginationDto, paginate } from '../common/dto/pagination.dto';

/**
 * EquipmentService
 * Handles industrial machine registration and management for clients.
 * Equipment is soft-deleted — isActive = false instead of hard delete.
 * Deactivated equipment cannot have new claims submitted against it.
 */
@Injectable()
export class EquipmentService {
  constructor(private readonly prisma: PrismaService) { }

  /**
   * Registers a new industrial machine for the authenticated client.
   * Serial number must be globally unique across all clients.
   * Commission date must be in the past.
   */
  async create(ownerId: string, dto: CreateEquipmentDto) {
    const commissionDate = new Date(dto.commissionDate);
    if (commissionDate > new Date()) {
      throw new BadRequestException('commissionDate cannot be in the future');
    }

    const exists = await this.prisma.equipment.findUnique({
      where: { serialNumber: dto.serialNumber },
    });
    if (exists) throw new ConflictException('Serial number already registered');

    return this.prisma.equipment.create({
      data: {
        ownerId,
        name: dto.name,
        type: dto.type,
        manufacturer: dto.manufacturer,
        model: dto.model,
        serialNumber: dto.serialNumber,
        commissionDate,
        location: dto.location,
      },
    });
  }

  /**
   * Returns a paginated list of active machines owned by this client.
   * Supports filtering by type and free-text search on name, manufacturer, model.
   */
  async findAllForOwner(
    ownerId: string,
    pagination: PaginationDto,
    type?: string,
    search?: string,
  ) {
    const { page = 1, limit = 10 } = pagination;
    const skip = (page - 1) * limit;

    const where: any = { ownerId, isActive: true };

    if (type) {
      where.type = type;
    }

    if (search) {
      where.OR = [
        { name: { contains: search, mode: 'insensitive' } },
        { manufacturer: { contains: search, mode: 'insensitive' } },
        { model: { contains: search, mode: 'insensitive' } },
        { serialNumber: { contains: search, mode: 'insensitive' } },
      ];
    }

    const [data, total] = await Promise.all([
      this.prisma.equipment.findMany({
        where,
        skip,
        take: limit,
        orderBy: { createdAt: 'desc' },
        include: {
          _count: { select: { claims: true } },
        },
      }),
      this.prisma.equipment.count({ where }),
    ]);

    return paginate(data, total, page, limit);
  }

  /**
   * Returns full details of one machine including recent claims.
   * Validates that the machine belongs to the requesting client.
   */
  async findOne(id: string, ownerId: string) {
    const equipment = await this.prisma.equipment.findUnique({
      where: { id },
      include: {
        claims: {
          orderBy: { createdAt: 'desc' },
          take: 5,
          select: {
            id: true,
            reference: true,
            status: true,
            claimedAmount: true,
            createdAt: true,
          },
        },
        _count: { select: { claims: true } },
      },
    });

    if (!equipment) throw new NotFoundException('Equipment not found');
    if (equipment.ownerId !== ownerId) throw new ForbiddenException('Not your equipment');

    return equipment;
  }

  /**
   * Updates machine information.
   * Serial number cannot be changed after registration.
   * commissionDate and lastMaintenanceDate must be converted from
   * ISO date strings to Date objects before passing to Prisma.
   */
  async update(id: string, ownerId: string, dto: UpdateEquipmentDto) {
    await this.findOne(id, ownerId);

    if (dto.commissionDate) {
      const commissionDate = new Date(dto.commissionDate);
      if (commissionDate > new Date()) {
        throw new BadRequestException('commissionDate cannot be in the future');
      }
    }

    if (dto.lastMaintenanceDate) {
      const lastMaintenance = new Date(dto.lastMaintenanceDate);
      if (lastMaintenance > new Date()) {
        throw new BadRequestException('lastMaintenanceDate cannot be in the future');
      }
    }

    return this.prisma.equipment.update({
      where: { id },
      data: {
        ...(dto.name && { name: dto.name }),
        ...(dto.location && { location: dto.location }),
        ...(dto.manufacturer && { manufacturer: dto.manufacturer }),
        ...(dto.model && { model: dto.model }),
        ...(dto.isActive !== undefined && { isActive: dto.isActive }),
        ...(dto.commissionDate && {
          commissionDate: new Date(dto.commissionDate),
        }),
        ...(dto.lastMaintenanceDate && {
          lastMaintenanceDate: new Date(dto.lastMaintenanceDate),
        }),
      },
    });
  }

  /**
   * Soft deletes a machine by setting isActive = false.
   * The machine record is preserved for historical claim records.
   */
  async remove(id: string, ownerId: string) {
    await this.findOne(id, ownerId);

    return this.prisma.equipment.update({
      where: { id },
      data: { isActive: false },
    });
  }

  /**
   * Verifies that a machine exists, is active, and belongs to the given client.
   * Used by ClaimsService before allowing a claim submission.
   */
  async verifyActiveAndOwned(equipmentId: string, ownerId: string) {
    const equipment = await this.prisma.equipment.findUnique({
      where: { id: equipmentId },
    });

    if (!equipment) throw new NotFoundException('Equipment not found');
    if (equipment.ownerId !== ownerId) throw new ForbiddenException('Not your equipment');
    if (!equipment.isActive) {
      throw new BadRequestException('Cannot submit a claim for deactivated equipment');
    }

    return equipment;
  }
}