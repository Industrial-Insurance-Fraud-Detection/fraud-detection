import { Controller, Get } from '@nestjs/common';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import {
    HealthCheckService,
    HealthCheck,
    MemoryHealthIndicator,
} from '@nestjs/terminus';
import { PrismaService } from '../prisma/prisma.service';
import { RedisService } from '../common/services/redis.service';

/**
 * HealthController
 * Exposes GET /api/v1/health for monitoring.
 * Checks PostgreSQL, Redis, and memory usage.
 * No authentication required.
 */
@ApiTags('Health')
@Controller('health')
export class HealthController {
    constructor(
        private readonly health: HealthCheckService,
        private readonly memory: MemoryHealthIndicator,
        private readonly prisma: PrismaService,
        private readonly redis: RedisService,
    ) { }

    @Get()
    @HealthCheck()
    @ApiOperation({ summary: 'Check all service connections and memory' })
    async check() {
        return this.health.check([
            // PostgreSQL
            async () => {
                try {
                    await this.prisma.$queryRaw`SELECT 1`;
                    return { postgresql: { status: 'up' } };
                } catch {
                    return { postgresql: { status: 'down' } };
                }
            },
            // Redis
            async () => {
                try {
                    await this.redis.set('health:ping', 'pong', 5);
                    return { redis: { status: 'up' } };
                } catch {
                    return { redis: { status: 'down' } };
                }
            },
            // memory heap — alert if over 300MB
            () => this.memory.checkHeap('memory_heap', 300 * 1024 * 1024),
        ]);
    }
}