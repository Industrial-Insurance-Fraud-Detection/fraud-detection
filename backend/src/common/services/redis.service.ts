import { Injectable, OnModuleInit, OnModuleDestroy, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';

/**
 * RedisService
 * Global service for all Redis operations.
 * Handles sessions, token blacklisting, and password reset tokens.
 * Compatible with Redis 3.x — sessions stored as JSON strings instead of hashes.
 */
@Injectable()
export class RedisService implements OnModuleInit, OnModuleDestroy {
    private client: Redis;
    private readonly logger = new Logger(RedisService.name);

    constructor(private config: ConfigService) { }

    /**
     * Initialize Redis connection on module load.
     * Reads host and port from environment variables.
     */
    onModuleInit() {
        this.client = new Redis({
            host: this.config.get<string>('REDIS_HOST') || 'localhost',
            port: this.config.get<number>('REDIS_PORT') || 6379,
        });

        this.client.on('connect', () => {
            this.logger.log('Redis connected successfully');
        });

        this.client.on('error', (err) => {
            this.logger.error('Redis connection error', err);
        });
    }

    /**
     * Gracefully close Redis connection on module destroy.
     */
    onModuleDestroy() {
        this.client.quit();
    }

    // ----------------------------------------------------------------
    // Core Key-Value Operations
    // ----------------------------------------------------------------

    /**
     * Store a string value in Redis.
     * @param key - Redis key
     * @param value - String value to store
     * @param ttlSeconds - Optional expiration time in seconds
     */
    async set(key: string, value: string, ttlSeconds?: number): Promise<void> {
        if (ttlSeconds) {
            await this.client.set(key, value, 'EX', ttlSeconds);
        } else {
            await this.client.set(key, value);
        }
    }

    /**
     * Retrieve a string value by key.
     * Returns null if key does not exist.
     * @param key - Redis key
     */
    async get(key: string): Promise<string | null> {
        return this.client.get(key);
    }

    /**
     * Delete a key from Redis.
     * @param key - Redis key to delete
     */
    async del(key: string): Promise<void> {
        await this.client.del(key);
    }

    /**
     * Check if a key exists in Redis.
     * @param key - Redis key to check
     * @returns true if key exists, false otherwise
     */
    async exists(key: string): Promise<boolean> {
        const result = await this.client.exists(key);
        return result === 1;
    }

    /**
     * Get the remaining time-to-live of a key in seconds.
     * Returns -1 if key has no expiration, -2 if key does not exist.
     * @param key - Redis key
     */
    async ttl(key: string): Promise<number> {
        return this.client.ttl(key);
    }

    /**
     * Delete all keys matching a pattern.
     * Used to delete all sessions for a specific user.
     * @param pattern - Redis key pattern, e.g. "session:userId:*"
     */
    async delPattern(pattern: string): Promise<void> {
        const keys = await this.client.keys(pattern);
        if (keys.length > 0) {
            await this.client.del(...keys);
        }
    }

    // ----------------------------------------------------------------
    // Session Operations (JSON string — Redis 3.x compatible)
    // ----------------------------------------------------------------

    /**
     * Store a session as a JSON string with expiration.
     * Uses plain SET instead of HSET for Redis 3.x compatibility.
     * @param sessionKey - Redis key for this session
     * @param data - Session data object
     * @param ttlSeconds - Session lifetime in seconds
     */
    async setSession(
        sessionKey: string,
        data: object,
        ttlSeconds: number,
    ): Promise<void> {
        await this.client.set(sessionKey, JSON.stringify(data), 'EX', ttlSeconds);
    }

    /**
     * Retrieve a session by key.
     * Returns null if session does not exist or has expired.
     * @param sessionKey - Redis key for the session
     */
    async getSession(sessionKey: string): Promise<Record<string, string> | null> {
        const data = await this.client.get(sessionKey);
        if (!data) return null;
        try {
            return JSON.parse(data);
        } catch {
            return null;
        }
    }

    /**
     * Update a single field within an existing session.
     * Reads the current session, updates the field, and writes it back.
     * @param sessionKey - Redis key for the session
     * @param field - Field name to update
     * @param value - New value for the field
     */
    async updateSession(
        sessionKey: string,
        field: string,
        value: string,
    ): Promise<void> {
        const existing = await this.getSession(sessionKey);
        if (existing) {
            existing[field] = value;
            await this.client.set(sessionKey, JSON.stringify(existing));
        }
    }
}