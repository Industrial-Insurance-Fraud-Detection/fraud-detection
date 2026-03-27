import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Readable } from 'stream';
import * as Minio from 'minio';

@Injectable()
export class MinioService implements OnModuleInit {
  private client: Minio.Client;
  private bucket: string;
  private readonly logger = new Logger(MinioService.name);

  constructor(private config: ConfigService) {
    this.bucket = this.config.get<string>('MINIO_BUCKET') || 'taamine-files';

    this.client = new Minio.Client({
      endPoint: this.config.get<string>('MINIO_ENDPOINT') || 'localhost',
      port: parseInt(this.config.get<string>('MINIO_PORT') || '9000'),
      useSSL: this.config.get<string>('MINIO_USE_SSL') === 'true',
      accessKey: this.config.get<string>('MINIO_ACCESS_KEY'),
      secretKey: this.config.get<string>('MINIO_SECRET_KEY'),
    });
  }

  async onModuleInit() {
    const exists = await this.client.bucketExists(this.bucket);
    if (!exists) {
      await this.client.makeBucket(this.bucket, 'us-east-1');
      this.logger.log(`✅ MinIO bucket "${this.bucket}" created`);
    } else {
      this.logger.log(`✅ MinIO bucket "${this.bucket}" exists`);
    }
  }

  /**
   * Upload a multipart file (from an HTTP request) to MinIO.
   * Returns the object path (stored in ClaimFile.minioPath).
   */
  async upload(
    claimId: string,
    file: Express.Multer.File,
  ): Promise<string> {
    const objectName = `claims/${claimId}/${Date.now()}-${file.originalname}`;
    await this.client.putObject(
      this.bucket,
      objectName,
      file.buffer,
      file.size,
      { 'Content-Type': file.mimetype },
    );
    return objectName;
  }

  /**
   * Upload a raw Buffer directly to MinIO under a given object name.
   * Used internally for generated PDFs and other programmatic uploads.
   * @param objectName  - Full MinIO object path (e.g. claims/id/decisions/file.pdf)
   * @param buffer      - File content as a Buffer
   * @param contentType - MIME type (e.g. 'application/pdf')
   */
  async uploadBuffer(
    objectName: string,
    buffer: Buffer,
    contentType: string,
  ): Promise<void> {
    const readable = Readable.from(buffer);
    await this.client.putObject(
      this.bucket,
      objectName,
      readable,
      buffer.length,
      { 'Content-Type': contentType },
    );
  }

  /**
   * Generate a 15-minute presigned GET URL for a file.
   */
  async getPresignedUrl(objectName: string): Promise<string> {
    return this.client.presignedGetObject(this.bucket, objectName, 15 * 60);
  }

  /**
   * Delete a file from MinIO.
   */
  async delete(objectName: string): Promise<void> {
    await this.client.removeObject(this.bucket, objectName);
  }
}