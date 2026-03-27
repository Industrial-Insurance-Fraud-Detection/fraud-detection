import { Injectable, NotFoundException, Logger } from '@nestjs/common';
import { Readable } from 'stream';
import { PrismaService } from '../prisma/prisma.service';
import { MinioService } from './minio.service';
import * as htmlPdf from 'html-pdf-node';

/**
 * FilesService
 * Handles file access control, presigned URL generation,
 * and PDF generation for decision letters.
 */
@Injectable()
export class FilesService {
    private readonly logger = new Logger(FilesService.name);

    constructor(
        private readonly prisma: PrismaService,
        private readonly minio: MinioService,
    ) { }

    /**
     * Returns a 15-minute presigned download URL for a claim file.
     * CLIENT can only access files belonging to their own claims.
     * INVESTIGATOR can access any file.
     */
    async getFileUrl(fileId: string, userId: string, userRole: string) {
        const file = await this.prisma.claimFile.findUnique({
            where: { id: fileId },
            include: {
                claim: {
                    select: { clientId: true, reference: true },
                },
            },
        });

        if (!file) throw new NotFoundException('File not found');

        if (userRole === 'CLIENT' && file.claim.clientId !== userId) {
            throw new NotFoundException('File not found');
        }

        const url = await this.minio.getPresignedUrl(file.minioPath);

        this.logger.log(
            `Presigned URL generated for file ${file.fileName} — claim ${file.claim.reference}`,
        );

        return {
            url,
            fileName: file.fileName,
            fileType: file.fileType,
            fileSize: file.fileSize,
            expiresIn: '15 minutes',
        };
    }

    /**
     * Converts an HTML string to a PDF buffer using html-pdf-node.
     * Called internally for PDF generation.
     */
    async generatePdfFromHtml(html: string): Promise<Buffer> {
        const file = { content: html };
        const options = {
            format: 'A4',
            margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
            printBackground: true,
        };

        this.logger.log('Generating PDF from HTML...');

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const pdfBuffer = await (htmlPdf as any).generatePdf(file, options);

        this.logger.log(`PDF generated — size: ${pdfBuffer.length} bytes`);

        return pdfBuffer as Buffer;
    }

    /**
     * Generates a PDF decision letter from HTML and saves it directly to MinIO.
     * Called by n8n after a claim is APPROVED or REJECTED.
     * Returns the MinIO path and a presigned download URL.
     */
    async generateAndSavePdf(
        html: string,
        claimId: string,
        fileName: string,
    ): Promise<{ minioPath: string; url: string; fileName: string }> {
        // generate PDF buffer
        const pdfBuffer = await this.generatePdfFromHtml(html);

        // build MinIO object path
        const objectName = `claims/${claimId}/decisions/${Date.now()}-${fileName}`;

        // upload via the public MinioService method — no private property access
        await this.minio.uploadBuffer(objectName, pdfBuffer, 'application/pdf');

        this.logger.log(`Decision PDF saved to MinIO: ${objectName}`);

        // generate presigned URL for immediate access
        const url = await this.minio.getPresignedUrl(objectName);

        return {
            minioPath: objectName,
            url,
            fileName,
        };
    }
}