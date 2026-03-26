import {
  Controller,
  Get,
  Post,
  Patch,
  Body,
  Param,
  Query,
  UseGuards,
  UseInterceptors,
  UploadedFiles,
} from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import {
  ApiTags,
  ApiOperation,
  ApiBearerAuth,
  ApiConsumes,
  ApiResponse,
  ApiQuery,
} from '@nestjs/swagger';
import { Role } from '@prisma/client';
import { ClaimsService } from './claims.service';
import { CreateClaimDto } from './dto/create-claim.dto';
import { DecideClaimDto } from './dto/decide-claim.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { PaginationDto } from '../common/dto/pagination.dto';

/**
 * ClaimsController
 * Handles all claim lifecycle endpoints.
 * CLIENT endpoints: submit, list own claims, view detail.
 * INVESTIGATOR endpoints: flagged queue, view detail, submit decision.
 * n8n endpoints: save PDF URL after decision letter generation.
 */
@ApiTags('Claims')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard, RolesGuard)
@Controller('claims')
export class ClaimsController {
  constructor(private readonly claimsService: ClaimsService) { }

  /**
   * POST /claims
   * CLIENT submits a new claim with supporting files.
   */
  @Post()
  @Roles(Role.CLIENT)
  @ApiOperation({ summary: 'Submit a new claim with files (CSV + photos + PDF)' })
  @ApiConsumes('multipart/form-data')
  @ApiResponse({ status: 201, description: 'Claim submitted and queued for AI analysis' })
  @ApiResponse({ status: 400, description: 'Validation error or missing required files' })
  @ApiResponse({ status: 403, description: 'Equipment does not belong to this client' })
  @ApiResponse({ status: 404, description: 'Equipment not found' })
  @UseInterceptors(FilesInterceptor('files', 20))
  submitClaim(
    @CurrentUser() user: any,
    @Body() dto: CreateClaimDto,
    @UploadedFiles() files: Express.Multer.File[],
  ) {
    return this.claimsService.submitClaim(user.id, dto, files || []);
  }

  /**
   * GET /claims/my
   * CLIENT gets their own claims with pagination.
   */
  @Get('my')
  @Roles(Role.CLIENT)
  @ApiOperation({ summary: 'List my claims with pagination' })
  @ApiQuery({ name: 'page', required: false, type: Number })
  @ApiQuery({ name: 'limit', required: false, type: Number })
  @ApiResponse({ status: 200, description: 'Paginated claims returned successfully' })
  findMyClaims(
    @CurrentUser() user: any,
    @Query() pagination: PaginationDto,
  ) {
    return this.claimsService.findMyClaims(user.id, pagination);
  }

  /**
   * GET /claims/flagged
   * INVESTIGATOR gets all claims awaiting human review.
   */
  @Get('flagged')
  @Roles(Role.INVESTIGATOR)
  @ApiOperation({ summary: 'Get flagged claims queue sorted by fraud score (INVESTIGATOR)' })
  @ApiQuery({ name: 'page', required: false, type: Number })
  @ApiQuery({ name: 'limit', required: false, type: Number })
  @ApiResponse({ status: 200, description: 'Paginated flagged claims returned' })
  @ApiResponse({ status: 403, description: 'Clients cannot access this endpoint' })
  getFlaggedClaims(@Query() pagination: PaginationDto) {
    return this.claimsService.getFlaggedClaims(pagination);
  }

  /**
   * GET /claims/:id
   * Returns full claim detail.
   * CLIENT can only view their own claims.
   * INVESTIGATOR can view any claim.
   */
  @Get(':id')
  @ApiOperation({ summary: 'Get full claim detail' })
  @ApiResponse({ status: 200, description: 'Claim returned successfully' })
  @ApiResponse({ status: 404, description: 'Claim not found' })
  findOne(@Param('id') id: string, @CurrentUser() user: any) {
    return this.claimsService.findOne(id, user.id, user.role);
  }

  /**
   * PATCH /claims/:id/decide
   * INVESTIGATOR submits APPROVED or REJECTED with mandatory notes.
   */
  @Patch(':id/decide')
  @Roles(Role.INVESTIGATOR)
  @ApiOperation({ summary: 'Submit human decision on a flagged claim (INVESTIGATOR)' })
  @ApiResponse({ status: 200, description: 'Decision recorded and client notified' })
  @ApiResponse({ status: 400, description: 'Claim is not in HUMAN_REVIEW status' })
  @ApiResponse({ status: 403, description: 'Clients cannot submit decisions' })
  @ApiResponse({ status: 404, description: 'Claim not found' })
  submitDecision(
    @Param('id') id: string,
    @CurrentUser() user: any,
    @Body() dto: DecideClaimDto,
  ) {
    return this.claimsService.submitDecision(id, user.id, dto);
  }

  /**
   * PATCH /claims/:id/pdf-url
   * Called by n8n to store the decision letter PDF URL on the claim.
   * No role restriction — called by internal service with JWT token.
   */
  @Patch(':id/pdf-url')
  @ApiOperation({ summary: 'Store decision PDF URL on claim (called by n8n)' })
  @ApiResponse({ status: 200, description: 'PDF URL saved successfully' })
  savePdfUrl(
    @Param('id') id: string,
    @Body() body: { pdfUrl: string },
  ) {
    return this.claimsService.savePdfUrl(id, body.pdfUrl);
  }
}