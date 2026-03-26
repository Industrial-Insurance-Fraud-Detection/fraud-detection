import { Global, Module } from '@nestjs/common';
import { RedisService } from './services/redis.service';
import { AuditService } from './services/audit.service';

/**
 * CommonModule
 * Global module that provides shared services across the entire application.
 * Marked as Global so no other module needs to import it explicitly.
 * Just import CommonModule once in AppModule and all services are available everywhere.
 */
@Global()
@Module({
    providers: [RedisService, AuditService],
    exports: [RedisService, AuditService],
})
export class CommonModule { }