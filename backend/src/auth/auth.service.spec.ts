import { Test, TestingModule } from '@nestjs/testing';
import { AuthService } from './auth.service';
import { PrismaService } from '../prisma/prisma.service';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { RedisService } from '../common/services/redis.service';
import { AuditService } from '../common/services/audit.service';
import { ConflictException, UnauthorizedException, BadRequestException } from '@nestjs/common';
import { Role } from '@prisma/client';
import * as bcrypt from 'bcryptjs';

/*
 * Mock all external dependencies so tests run without
 * a real database, Redis, or any external service.
 * Each mock provides only the methods used by AuthService.
 */

const mockPrismaService = {
    user: {
        findUnique: jest.fn(),
        create: jest.fn(),
        update: jest.fn(),
    },
};

const mockRedisService = {
    exists: jest.fn(),
    get: jest.fn(),
    set: jest.fn(),
    del: jest.fn(),
    delPattern: jest.fn(),
    ttl: jest.fn(),
    setSession: jest.fn(),
    getSession: jest.fn(),
};

const mockJwtService = {
    sign: jest.fn(),
    verify: jest.fn(),
};

const mockConfigService = {
    get: jest.fn((key: string) => {
        /*
         * Return test values for all environment variables
         * used by AuthService.
         */
        const config: Record<string, string> = {
            JWT_ACCESS_SECRET: 'test-access-secret',
            JWT_REFRESH_SECRET: 'test-refresh-secret',
            JWT_ACCESS_EXPIRATION: '15m',
            JWT_REFRESH_EXPIRATION: '7d',
            BCRYPT_ROUNDS: '10',
            MAX_LOGIN_ATTEMPTS: '5',
            LOCKOUT_DURATION_MINUTES: '30',
        };
        return config[key];
    }),
};

const mockAuditService = {
    /*
     * Audit logging is a side effect — we mock it to do nothing
     * so tests focus on the main auth logic only.
     */
    log: jest.fn().mockResolvedValue(undefined),
};

/*
 * Reusable test data used across multiple test cases.
 */
const TEST_IP = '127.0.0.1';
const TEST_UA = 'jest-test-agent';
const TEST_USER_ID = 'user-id-123';
const TEST_EMAIL = 'ahmed@sonatrach.dz';
const TEST_PASSWORD = 'StrongPass123';
const TEST_JTI = 'jti-uuid-123';

describe('AuthService', () => {
    let service: AuthService;

    beforeEach(async () => {
        /*
         * Create a fresh NestJS testing module before each test.
         * All real providers are replaced with mocks.
         */
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AuthService,
                { provide: PrismaService, useValue: mockPrismaService },
                { provide: JwtService, useValue: mockJwtService },
                { provide: ConfigService, useValue: mockConfigService },
                { provide: RedisService, useValue: mockRedisService },
                { provide: AuditService, useValue: mockAuditService },
            ],
        }).compile();

        service = module.get<AuthService>(AuthService);

        /*
         * Reset all mocks before each test to prevent
         * state leaking between test cases.
         */
        jest.clearAllMocks();
    });

    // ----------------------------------------------------------------
    // Register
    // ----------------------------------------------------------------

    describe('register', () => {
        const registerDto = {
            firstName: 'Ahmed',
            lastName: 'Benali',
            email: TEST_EMAIL,
            password: TEST_PASSWORD,
            companyName: 'Sonatrach',
            phone: '0550000001',
            wilaya: 'Boumerdes',
        };

        it('should register a new client successfully and return tokens', async () => {
            /*
             * No existing user found — registration can proceed.
             */
            mockPrismaService.user.findUnique.mockResolvedValue(null);
            mockPrismaService.user.create.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                firstName: 'Ahmed',
                lastName: 'Benali',
            });
            mockJwtService.sign.mockReturnValue('mock-token');
            mockRedisService.setSession.mockResolvedValue(undefined);

            const result = await service.register(registerDto, TEST_IP, TEST_UA);

            expect(result.user.email).toBe(TEST_EMAIL);
            expect(result.user.role).toBe(Role.CLIENT);
            expect(result.accessToken).toBeDefined();
            expect(result.refreshToken).toBeDefined();
        });

        it('should throw ConflictException if email is already registered', async () => {
            /*
             * Simulate an existing user with the same email.
             */
            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
            });

            await expect(
                service.register(registerDto, TEST_IP, TEST_UA),
            ).rejects.toThrow(ConflictException);
        });

        it('should hash the password before saving to database', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);
            mockPrismaService.user.create.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                firstName: 'Ahmed',
                lastName: 'Benali',
            });
            mockJwtService.sign.mockReturnValue('mock-token');
            mockRedisService.setSession.mockResolvedValue(undefined);

            await service.register(registerDto, TEST_IP, TEST_UA);

            /*
             * Verify that the password saved to the database
             * is a bcrypt hash, not the original plain text.
             */
            const createCall = mockPrismaService.user.create.mock.calls[0][0];
            const savedHash = createCall.data.passwordHash;
            const hashMatches = await bcrypt.compare(TEST_PASSWORD, savedHash);

            expect(savedHash).not.toBe(TEST_PASSWORD);
            expect(hashMatches).toBe(true);
        });

        it('should always assign CLIENT role regardless of input', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);
            mockPrismaService.user.create.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                firstName: 'Ahmed',
                lastName: 'Benali',
            });
            mockJwtService.sign.mockReturnValue('mock-token');
            mockRedisService.setSession.mockResolvedValue(undefined);

            await service.register(registerDto, TEST_IP, TEST_UA);

            const createCall = mockPrismaService.user.create.mock.calls[0][0];
            expect(createCall.data.role).toBe('CLIENT');
        });
    });

    // ----------------------------------------------------------------
    // Login
    // ----------------------------------------------------------------

    describe('login', () => {
        const loginDto = {
            email: TEST_EMAIL,
            password: TEST_PASSWORD,
        };

        it('should login successfully and return tokens', async () => {
            const hashedPassword = await bcrypt.hash(TEST_PASSWORD, 10);

            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                firstName: 'Ahmed',
                lastName: 'Benali',
                passwordHash: hashedPassword,
            });
            mockRedisService.exists.mockResolvedValue(false);
            mockRedisService.del.mockResolvedValue(undefined);
            mockJwtService.sign.mockReturnValue('mock-token');
            mockRedisService.setSession.mockResolvedValue(undefined);

            const result = await service.login(loginDto, TEST_IP, TEST_UA);

            expect(result.user.email).toBe(TEST_EMAIL);
            expect(result.accessToken).toBeDefined();
            expect(result.refreshToken).toBeDefined();
        });

        it('should throw UnauthorizedException if user is not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            await expect(
                service.login(loginDto, TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if password is wrong', async () => {
            const hashedPassword = await bcrypt.hash('DifferentPass123', 10);

            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                passwordHash: hashedPassword,
            });
            mockRedisService.exists.mockResolvedValue(false);
            mockRedisService.get.mockResolvedValue(null);
            mockRedisService.set.mockResolvedValue(undefined);

            await expect(
                service.login(loginDto, TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);
        });

        it('should throw UnauthorizedException if account is locked', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                passwordHash: 'some-hash',
            });

            /*
             * Simulate account lockout — lockout key exists in Redis.
             */
            mockRedisService.exists.mockResolvedValue(true);
            mockRedisService.ttl.mockResolvedValue(1200);

            await expect(
                service.login(loginDto, TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);
        });

        it('should lock account after max failed attempts', async () => {
            const hashedPassword = await bcrypt.hash('DifferentPass123', 10);

            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                passwordHash: hashedPassword,
            });
            mockRedisService.exists.mockResolvedValue(false);

            /*
             * Simulate 4 previous failed attempts — next one triggers lockout.
             */
            mockRedisService.get.mockResolvedValue('4');
            mockRedisService.set.mockResolvedValue(undefined);

            await expect(
                service.login(loginDto, TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);

            /*
             * Verify that the lockout key was set in Redis.
             * Two set calls expected: one for attempts counter, one for lockout key.
             */
            expect(mockRedisService.set).toHaveBeenCalledTimes(2);
        });

        it('should not expose whether email exists when user is not found', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            try {
                await service.login(loginDto, TEST_IP, TEST_UA);
            } catch (error) {
                /*
                 * Error message must be identical for both
                 * "user not found" and "wrong password" cases.
                 */
                expect(error.message).toBe('Invalid credentials');
            }
        });
    });

    // ----------------------------------------------------------------
    // Refresh Token
    // ----------------------------------------------------------------

    describe('refresh', () => {
        it('should rotate refresh token and return new token pair', async () => {
            mockJwtService.verify.mockReturnValue({
                sub: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                jti: TEST_JTI,
            });
            mockRedisService.getSession.mockResolvedValue({
                userId: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
            });
            mockRedisService.del.mockResolvedValue(undefined);
            mockJwtService.sign.mockReturnValue('new-mock-token');
            mockRedisService.setSession.mockResolvedValue(undefined);

            const result = await service.refresh('valid-refresh-token', TEST_IP, TEST_UA);

            expect(result.accessToken).toBeDefined();
            expect(result.refreshToken).toBeDefined();

            /*
             * Verify old session was deleted before new one was created.
             */
            expect(mockRedisService.del).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:${TEST_JTI}`,
            );
        });

        it('should revoke all sessions when refresh token reuse is detected', async () => {
            mockJwtService.verify.mockReturnValue({
                sub: TEST_USER_ID,
                email: TEST_EMAIL,
                role: Role.CLIENT,
                jti: TEST_JTI,
            });

            /*
             * Session not found in Redis — token was already used.
             * This indicates a reuse attack.
             */
            mockRedisService.getSession.mockResolvedValue(null);
            mockRedisService.delPattern.mockResolvedValue(undefined);

            await expect(
                service.refresh('reused-refresh-token', TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);

            /*
             * All sessions for this user must be revoked.
             */
            expect(mockRedisService.delPattern).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:*`,
            );
        });

        it('should throw UnauthorizedException if refresh token is invalid', async () => {
            mockJwtService.verify.mockImplementation(() => {
                throw new Error('invalid token');
            });

            await expect(
                service.refresh('invalid-token', TEST_IP, TEST_UA),
            ).rejects.toThrow(UnauthorizedException);
        });
    });

    // ----------------------------------------------------------------
    // Logout
    // ----------------------------------------------------------------

    describe('logout', () => {
        it('should blacklist access token and delete session on logout', async () => {
            mockRedisService.set.mockResolvedValue(undefined);
            mockJwtService.verify.mockReturnValue({ jti: 'refresh-jti-123' });
            mockRedisService.del.mockResolvedValue(undefined);

            const result = await service.logout(
                TEST_USER_ID,
                'access-token',
                'refresh-token',
                TEST_JTI,
                TEST_IP,
                TEST_UA,
            );

            expect(result.message).toBe('Logged out successfully');

            /*
             * Access token jti must be added to the blacklist.
             */
            expect(mockRedisService.set).toHaveBeenCalledWith(
                `blacklist:${TEST_JTI}`,
                '1',
                expect.any(Number),
            );

            /*
             * Refresh token session must be deleted from Redis.
             */
            expect(mockRedisService.del).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:refresh-jti-123`,
            );
        });
    });

    // ----------------------------------------------------------------
    // Logout All
    // ----------------------------------------------------------------

    describe('logoutAll', () => {
        it('should blacklist current token and delete all user sessions', async () => {
            mockRedisService.set.mockResolvedValue(undefined);
            mockRedisService.delPattern.mockResolvedValue(undefined);

            const result = await service.logoutAll(
                TEST_USER_ID,
                TEST_JTI,
                TEST_IP,
                TEST_UA,
            );

            expect(result.message).toBe('Logged out from all devices successfully');

            expect(mockRedisService.set).toHaveBeenCalledWith(
                `blacklist:${TEST_JTI}`,
                '1',
                expect.any(Number),
            );

            expect(mockRedisService.delPattern).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:*`,
            );
        });
    });

    // ----------------------------------------------------------------
    // Change Password
    // ----------------------------------------------------------------

    describe('changePassword', () => {
        const changePasswordDto = {
            currentPassword: TEST_PASSWORD,
            newPassword: 'NewStrongPass456',
        };

        it('should change password and revoke all sessions', async () => {
            const hashedPassword = await bcrypt.hash(TEST_PASSWORD, 10);

            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                passwordHash: hashedPassword,
            });
            mockPrismaService.user.update.mockResolvedValue({});
            mockRedisService.set.mockResolvedValue(undefined);
            mockRedisService.delPattern.mockResolvedValue(undefined);

            const result = await service.changePassword(
                TEST_USER_ID,
                TEST_JTI,
                changePasswordDto,
                TEST_IP,
                TEST_UA,
            );

            expect(result.message).toContain('Password changed successfully');

            /*
             * All sessions must be revoked after password change.
             */
            expect(mockRedisService.delPattern).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:*`,
            );
        });

        it('should throw BadRequestException if current password is wrong', async () => {
            const hashedPassword = await bcrypt.hash('DifferentPass123', 10);

            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                passwordHash: hashedPassword,
            });

            await expect(
                service.changePassword(
                    TEST_USER_ID,
                    TEST_JTI,
                    changePasswordDto,
                    TEST_IP,
                    TEST_UA,
                ),
            ).rejects.toThrow(BadRequestException);
        });
    });

    // ----------------------------------------------------------------
    // Forgot Password
    // ----------------------------------------------------------------

    describe('forgotPassword', () => {
        it('should return reset token when email exists', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue({
                id: TEST_USER_ID,
                email: TEST_EMAIL,
            });
            mockRedisService.set.mockResolvedValue(undefined);

            const result = await service.forgotPassword(
                { email: TEST_EMAIL },
                TEST_IP,
            );

            expect(result.message).toBeDefined();
            expect(result.resetToken).toBeDefined();
        });

        it('should return same message when email does not exist to prevent enumeration', async () => {
            mockPrismaService.user.findUnique.mockResolvedValue(null);

            const result = await service.forgotPassword(
                { email: 'unknown@test.dz' },
                TEST_IP,
            );

            /*
             * Must return success message even when email is not found.
             * This prevents attackers from discovering registered emails.
             */
            expect(result.message).toBeDefined();
            expect(result.resetToken).toBeUndefined();
        });
    });

    // ----------------------------------------------------------------
    // Reset Password
    // ----------------------------------------------------------------

    describe('resetPassword', () => {
        const resetDto = {
            token: 'valid-reset-token',
            newPassword: 'NewStrongPass456',
        };

        it('should reset password and revoke all sessions with valid token', async () => {
            /*
             * Simulate a valid reset token stored in Redis.
             */
            mockRedisService.get.mockResolvedValue(TEST_USER_ID);
            mockPrismaService.user.update.mockResolvedValue({});
            mockRedisService.del.mockResolvedValue(undefined);
            mockRedisService.delPattern.mockResolvedValue(undefined);

            const result = await service.resetPassword(resetDto, TEST_IP);

            expect(result.message).toContain('Password reset successfully');

            /*
             * Reset token must be deleted after use so it cannot be reused.
             */
            expect(mockRedisService.del).toHaveBeenCalledWith(
                `reset:${resetDto.token}`,
            );

            /*
             * All sessions must be revoked after password reset.
             */
            expect(mockRedisService.delPattern).toHaveBeenCalledWith(
                `session:${TEST_USER_ID}:*`,
            );
        });

        it('should throw BadRequestException if reset token is invalid or expired', async () => {
            /*
             * Simulate token not found in Redis — expired or never existed.
             */
            mockRedisService.get.mockResolvedValue(null);

            await expect(
                service.resetPassword(resetDto, TEST_IP),
            ).rejects.toThrow(BadRequestException);
        });
    });
});