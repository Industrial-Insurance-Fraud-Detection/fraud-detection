import { Controller, Get, Post, Param, Body, UseGuards, Res } from '@nestjs/common';
import {
  ApiTags,
  ApiOperation,
  ApiBearerAuth,
  ApiResponse,
} from '@nestjs/swagger';
import { Response } from 'express';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { FilesService } from './files.service';

/**
 * FilesController
 * Handles presigned URL generation for claim file downloads.
 * Also handles PDF generation and storage for decision letters.
 */
@ApiTags('Files')
@ApiBearerAuth()
@UseGuards(JwtAuthGuard)
@Controller('files')
export class FilesController {
  constructor(private readonly filesService: FilesService) { }

  /**
   * GET /files/:id/url
   * Returns a 15-minute presigned URL for downloading a claim file.
   * CLIENT can only access files from their own claims.
   * INVESTIGATOR can access any file.
   */
  @Get(':id/url')
  @ApiOperation({ summary: 'Get 15-minute presigned download URL for a file' })
  @ApiResponse({ status: 200, description: 'Presigned URL returned successfully' })
  @ApiResponse({ status: 404, description: 'File not found or access denied' })
  getFileUrl(@Param('id') id: string, @CurrentUser() user: any) {
    return this.filesService.getFileUrl(id, user.id, user.role);
  }

  /**
   * POST /files/generate-pdf
   * Called by n8n to convert HTML to PDF and return binary.
   * Protected by JWT — n8n sends a service token.
   */
  @Post('generate-pdf')
  @ApiOperation({ summary: 'Generate PDF from HTML string (called by n8n)' })
  @ApiResponse({ status: 200, description: 'PDF binary returned' })
  async generatePdf(
    @Body() body: { html: string; fileName: string },
    @Res() res: Response,
  ) {
    const pdfBuffer = await this.filesService.generatePdfFromHtml(body.html);
    res.set({
      'Content-Type': 'application/pdf',
      'Content-Disposition': `attachment; filename="${body.fileName}"`,
      'Content-Length': pdfBuffer.length,
    });
    res.end(pdfBuffer);
  }

  /**
   * POST /files/generate-and-save-pdf
   * Called by n8n to generate a PDF decision letter and save it to MinIO.
   * Returns the MinIO path and a presigned URL for immediate download.
   * This single endpoint replaces a separate generate + upload flow.
   */
  @Post('generate-and-save-pdf')
  @ApiOperation({ summary: 'Generate PDF and save to MinIO (called by n8n)' })
  @ApiResponse({ status: 201, description: 'PDF generated and saved, URL returned' })
  async generateAndSavePdf(
    @Body() body: { html: string; claimId: string; fileName: string },
  ) {
    return this.filesService.generateAndSavePdf(
      body.html,
      body.claimId,
      body.fileName,
    );
  }
}