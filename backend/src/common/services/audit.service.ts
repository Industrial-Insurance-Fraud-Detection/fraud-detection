import { Injectable, Logger } from '@nestjs/common';
import { PrismaService } from '../../prisma/prisma.service';

/**
 * Audit event types for all security-related actions.
 * Every auth event must use one of these constants.
 */
export const AuditEvent = {
    REGISTER: 'REGISTER',
    LOGIN_SUCCESS: 'LOGIN_SUCCESS',
    LOGIN_FAILED: 'LOGIN_FAILED',
    LOGOUT: 'LOGOUT',
    LOGOUT_ALL: 'LOGOUT_ALL',
    TOKEN_REFRESHED: 'TOKEN_REFRESHED',
    PASSWORD_CHANGED: 'PASSWORD_CHANGED',
    PASSWORD_RESET_REQUESTED: 'PASSWORD_RESET_REQUESTED',
    PASSWORD_RESET_SUCCESS: 'PASSWORD_RESET_SUCCESS',
    ACCOUNT_LOCKED: 'ACCOUNT_LOCKED',
} as const;

export type AuditEventType = typeof AuditEvent[keyof typeof AuditEvent];

/**
 * Payload required to create an audit log entry.
 */
export interface AuditLogPayload {
    userId?: string;
    event: AuditEventType;
    ip?: string;
    userAgent?: string;
    metadata?: Record<string, any>;
}

/**
 * AuditService
 * Records every security event to the audit_logs table.
 * Called after every auth action — login, logout, password change, etc.
 * Failures are logged but never throw, so they never break the main flow.
 */
@Injectable()
export class AuditService {
    private readonly logger = new Logger(AuditService.name);

    constructor(private prisma: PrismaService) { }

    /**
     * Create a new audit log entry.
     * This method never throws — errors are caught and logged silently
     * so that an audit failure never breaks authentication.
     * @param payload - Audit log data
     */
    async log(payload: AuditLogPayload): Promise<void> {
        try {
            await this.prisma.auditLog.create({
                data: {
                    userId: payload.userId || null,
                    event: payload.event,
                    ip: payload.ip || null,
                    userAgent: payload.userAgent || null,
                    metadata: payload.metadata || null,
                },
            });
        } catch (error) {
            // Log the failure but never throw.
            // Audit logging must never interrupt the auth flow.
            this.logger.error(
                `Failed to write audit log for event "${payload.event}": ${error.message}`,
            );
        }
    }

    /**
     * Retrieve audit logs for a specific user, most recent first.
     * Used by admin or investigator to review account activity.
     * @param userId - The user whose logs to retrieve
     * @param limit  - Maximum number of records to return (default 50)
     */
    async getLogsForUser(userId: string, limit = 50) {
        return this.prisma.auditLog.findMany({
            where: { userId },
            orderBy: { createdAt: 'desc' },
            take: limit,
        });
    }
}