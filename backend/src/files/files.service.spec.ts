import { Test, TestingModule } from '@nestjs/testing';
import { FilesService } from './files.service';
import { PrismaService } from '../prisma/prisma.service';
import { MinioService } from './minio.service';
import { NotFoundException } from '@nestjs/common';

/**
 * FilesService Tests
 * All Prisma and MinIO calls are mocked.
 */
describe('FilesService', () => {
    let service: FilesService;
    let prisma: any;
    let minio: any;

    const mockFile = {
        id: 'file-123',
        claimId: 'claim-123',
        fileType: 'PHOTO',
        minioPath: 'claims/claim-123/1234567890-damage.jpg',
        fileName: 'damage.jpg',
        fileSize: 204800,
        createdAt: new Date(),
        claim: {
            clientId: 'user-123',
            reference: 'SIN-2026-ABC123',
        },
    };

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                FilesService,
                {
                    provide: PrismaService,
                    useValue: {
                        claimFile: {
                            findUnique: jest.fn(),
                        },
                    },
                },
                {
                    provide: MinioService,
                    useValue: {
                        getPresignedUrl: jest.fn(),
                    },
                },
            ],
        }).compile();

        service = module.get<FilesService>(FilesService);
        prisma = module.get<PrismaService>(PrismaService);
        minio = module.get<MinioService>(MinioService);
    });

    afterEach(() => jest.clearAllMocks());

    // ─── getFileUrl ───────────────────────────────────────────────────────────

    describe('getFileUrl', () => {
        it('should return presigned URL for file owner', async () => {
            prisma.claimFile.findUnique.mockResolvedValue(mockFile);
            minio.getPresignedUrl.mockResolvedValue('https://minio/presigned-url');

            const result = await service.getFileUrl('file-123', 'user-123', 'CLIENT');

            expect(result.url).toBe('https://minio/presigned-url');
            expect(result.fileName).toBe('damage.jpg');
            expect(result.fileType).toBe('PHOTO');
            expect(result.fileSize).toBe(204800);
            expect(result.expiresIn).toBe('15 minutes');
        });

        it('should return presigned URL for investigator on any file', async () => {
            prisma.claimFile.findUnique.mockResolvedValue(mockFile);
            minio.getPresignedUrl.mockResolvedValue('https://minio/presigned-url');

            const result = await service.getFileUrl('file-123', 'inv-456', 'INVESTIGATOR');

            expect(result.url).toBe('https://minio/presigned-url');
        });

        it('should throw NotFoundException if file does not exist', async () => {
            prisma.claimFile.findUnique.mockResolvedValue(null);

            await expect(
                service.getFileUrl('nonexistent-id', 'user-123', 'CLIENT'),
            ).rejects.toThrow(NotFoundException);
        });

        it('should throw NotFoundException if client tries to access another users file', async () => {
            prisma.claimFile.findUnique.mockResolvedValue({
                ...mockFile,
                claim: { clientId: 'other-user-999', reference: 'SIN-2026-XYZ' },
            });

            await expect(
                service.getFileUrl('file-123', 'user-123', 'CLIENT'),
            ).rejects.toThrow(NotFoundException);

            // MinIO should never be called if access is denied
            expect(minio.getPresignedUrl).not.toHaveBeenCalled();
        });

        it('should not expose file existence to unauthorized clients', async () => {
            prisma.claimFile.findUnique.mockResolvedValue({
                ...mockFile,
                claim: { clientId: 'other-user-999', reference: 'SIN-2026-XYZ' },
            });

            // should throw 404 not 403 — does not reveal the file exists
            await expect(
                service.getFileUrl('file-123', 'user-123', 'CLIENT'),
            ).rejects.toThrow(NotFoundException);
        });
    });
});