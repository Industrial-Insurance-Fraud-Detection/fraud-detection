import { Module } from '@nestjs/common';
import { TerminusModule } from '@nestjs/terminus';
import { HealthController } from './health.controller';
import { CommonModule } from '../common/common.module';

/**
 * HealthModule
 * Provides the /health endpoint using @nestjs/terminus.
 * CommonModule provides RedisService.
 */
@Module({
    imports: [TerminusModule, CommonModule],
    controllers: [HealthController],
})
export class HealthModule { }