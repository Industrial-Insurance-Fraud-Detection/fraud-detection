import { Module } from '@nestjs/common';
import { FilesController } from './files.controller';
import { FilesService } from './files.service';
import { MinioService } from './minio.service';

/**
 * FilesModule
 * Provides MinioService to other modules via exports.
 * FilesService handles access control for file downloads.
 */
@Module({
  controllers: [FilesController],
  providers: [FilesService, MinioService],
  exports: [MinioService],
})
export class FilesModule { }