import {
  Controller,
  Get,
  Post,
  Patch,
  Delete,
  Body,
  Param,
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
import { Role } from '@prisma/client';
import { EquipmentService } from './equipment.service';
import { CreateEquipmentDto } from './dto/create-equipment.dto';
import { UpdateEquipmentDto } from './dto/update-equipment.dto';
import { EquipmentQueryDto } from './dto/equipment-query.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { CurrentUser } from '../common/decorators/current-user.decorator';

/**
 * EquipmentController
 * All endpoints are CLIENT only — investigators do not manage equipment.
 * Equipment belongs to the authenticated client — ownership is enforced in the service.
 */
@ApiTags('Equipment')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, RolesGuard)
@Roles(Role.CLIENT)
@Controller('equipment')
export class EquipmentController {
  constructor(private readonly equipmentService: EquipmentService) { }

  /**
   * POST /equipment
   * Registers a new industrial machine for the authenticated client.
   */
  @Post()
  @ApiOperation({ summary: 'Register a new industrial machine' })
  @ApiResponse({ status: 201, description: 'Machine registered successfully' })
  @ApiResponse({ status: 400, description: 'Validation error or future commission date' })
  @ApiResponse({ status: 409, description: 'Serial number already registered' })
  create(@CurrentUser() user: any, @Body() dto: CreateEquipmentDto) {
    return this.equipmentService.create(user.id, dto);
  }

  /**
   * GET /equipment
   * Returns a paginated list of active machines owned by this client.
   * Supports optional filtering by type and free-text search.
   * EquipmentQueryDto whitelists page, limit, search, and type so the
   * global forbidNonWhitelisted ValidationPipe does not reject them.
   */
  @Get()
  @ApiOperation({ summary: 'List my machines with pagination and search' })
  @ApiResponse({ status: 200, description: 'Paginated list returned successfully' })
  findAll(
    @CurrentUser() user: any,
    @Query() query: EquipmentQueryDto,
  ) {
    return this.equipmentService.findAllForOwner(
      user.id,
      { page: query.page, limit: query.limit },
      query.type,
      query.search,
    );
  }

  /**
   * GET /equipment/:id
   * Returns full details of one machine including last 5 claims.
   */
  @Get(':id')
  @ApiOperation({ summary: 'Get one machine with recent claims' })
  @ApiResponse({ status: 200, description: 'Machine returned successfully' })
  @ApiResponse({ status: 403, description: 'Not your equipment' })
  @ApiResponse({ status: 404, description: 'Equipment not found' })
  findOne(@Param('id') id: string, @CurrentUser() user: any) {
    return this.equipmentService.findOne(id, user.id);
  }

  /**
   * PATCH /equipment/:id
   * Updates machine information. Serial number cannot be changed.
   */
  @Patch(':id')
  @ApiOperation({ summary: 'Update machine info' })
  @ApiResponse({ status: 200, description: 'Machine updated successfully' })
  @ApiResponse({ status: 400, description: 'Validation error' })
  @ApiResponse({ status: 403, description: 'Not your equipment' })
  update(
    @Param('id') id: string,
    @CurrentUser() user: any,
    @Body() dto: UpdateEquipmentDto,
  ) {
    return this.equipmentService.update(id, user.id, dto);
  }

  /**
   * DELETE /equipment/:id
   * Soft deletes the machine — sets isActive = false.
   * Historical claim records are preserved.
   */
  @Delete(':id')
  @ApiOperation({ summary: 'Deactivate machine (soft delete)' })
  @ApiResponse({ status: 200, description: 'Machine deactivated successfully' })
  @ApiResponse({ status: 403, description: 'Not your equipment' })
  remove(@Param('id') id: string, @CurrentUser() user: any) {
    return this.equipmentService.remove(id, user.id);
  }
}