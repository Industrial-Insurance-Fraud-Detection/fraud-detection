import {
  Controller,
  Get,
  Patch,
  Param,
  Body,
  UseGuards,
  Request,
} from '@nestjs/common';
import {
  ApiBearerAuth,
  ApiOperation,
  ApiTags,
  ApiResponse,
} from '@nestjs/swagger';
import { UsersService } from './users.service';
import { UpdateUserDto } from './dto/update-user.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';

/**
 * UsersController
 * Handles profile management for authenticated users.
 * All endpoints require a valid JWT access token.
 */
@ApiTags('Users')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, RolesGuard)
@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) { }

  /**
   * GET /users/me
   * Returns the full profile of the currently authenticated user.
   * Available to both CLIENT and INVESTIGATOR roles.
   */
  @Get('me')
  @ApiOperation({ summary: 'Get my profile' })
  @ApiResponse({ status: 200, description: 'Profile returned successfully' })
  getMe(@Request() req) {
    return this.usersService.getMe(req.user.id);
  }

  /**
   * GET /users/me/stats
   * Returns claim statistics for the authenticated client.
   * Shows total, approved, rejected, pending counts and approval rate.
   */
  @Get('me/stats')
  @Roles('CLIENT')
  @ApiOperation({ summary: 'Get my claim statistics' })
  @ApiResponse({ status: 200, description: 'Statistics returned successfully' })
  @ApiResponse({ status: 403, description: 'Investigators cannot access this endpoint' })
  getMyStats(@Request() req) {
    return this.usersService.getMyStats(req.user.id);
  }

  /**
   * PATCH /users/me
   * Updates the profile of the currently authenticated user.
   * Role and email cannot be changed via this endpoint.
   */
  @Patch('me')
  @ApiOperation({ summary: 'Update my profile' })
  @ApiResponse({ status: 200, description: 'Profile updated successfully' })
  @ApiResponse({ status: 400, description: 'Validation error' })
  updateMe(@Request() req, @Body() dto: UpdateUserDto) {
    return this.usersService.updateMe(req.user.id, dto);
  }

  /**
   * GET /users/:id
   * Returns a client's public profile.
   * Only accessible by investigators when reviewing a claim.
   */
  @Get(':id')
  @Roles('INVESTIGATOR')
  @ApiOperation({ summary: 'Get a client profile (investigator only)' })
  @ApiResponse({ status: 200, description: 'Profile returned successfully' })
  @ApiResponse({ status: 403, description: 'Clients cannot access this endpoint' })
  @ApiResponse({ status: 404, description: 'User not found' })
  getUserById(@Request() req, @Param('id') id: string) {
    return this.usersService.getUserById(req.user.id, req.user.role, id);
  }
}