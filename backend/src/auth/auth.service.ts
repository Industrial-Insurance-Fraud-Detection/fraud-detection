import {
  Injectable,
  UnauthorizedException,
  ConflictException,
  BadRequestException,
  Logger,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { RedisService } from '../common/services/redis.service';
import { AuditService, AuditEvent } from '../common/services/audit.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { ChangePasswordDto } from './dto/change-password.dto';
import { ForgotPasswordDto } from './dto/forgot-password.dto';
import { ResetPasswordDto } from './dto/reset-password.dto';
import * as bcrypt from 'bcryptjs';
import { v4 as uuidv4 } from 'uuid';

/**
 * Token lifetime constants in seconds.
 * Used when storing sessions and blacklist entries in Redis.
 */
const TOKEN_TTL = {
  ACCESS: 15 * 60,           // 15 minutes
  REFRESH: 7 * 24 * 60 * 60,  // 7 days
  RESET: 15 * 60,           // 15 minutes
};

/**
 * Redis key prefixes.
 * Centralizing prefixes avoids typos and makes pattern deletion reliable.
 */
const REDIS_PREFIX = {
  SESSION: 'session',
  BLACKLIST: 'blacklist',
  RESET: 'reset',
  LOCKOUT: 'lockout',
  ATTEMPTS: 'attempts',
};

/**
 * AuthService
 * Handles all authentication logic including registration, login,
 * token rotation, logout, password change, and password reset.
 * Sessions are stored in Redis. All security events are audit logged.
 */
@Injectable()
export class AuthService {
  private readonly logger = new Logger(AuthService.name);

  /*
   * Maximum failed login attempts before account is locked.
   * Lockout duration is also configurable via environment variables.
   */
  private readonly maxAttempts: number;
  private readonly lockoutDuration: number;
  private readonly bcryptRounds: number;

  constructor(
    private prisma: PrismaService,
    private jwt: JwtService,
    private config: ConfigService,
    private redis: RedisService,
    private audit: AuditService,
  ) {
    this.maxAttempts = parseInt(config.get('MAX_LOGIN_ATTEMPTS') || '5');
    this.lockoutDuration = parseInt(config.get('LOCKOUT_DURATION_MINUTES') || '30') * 60;
    this.bcryptRounds = parseInt(config.get('BCRYPT_ROUNDS') || '12');
  }

  // ----------------------------------------------------------------
  // Register
  // ----------------------------------------------------------------

  /**
   * Register a new user account.
   * Hashes the password, saves the user, and returns tokens immediately.
   * No email verification required.
   * @param dto     - Registration data
   * @param ip      - Client IP address for audit log
   * @param userAgent - Client user agent for audit log
   */
  async register(dto: RegisterDto, ip: string, userAgent: string) {
    /*
     * Check if email is already registered.
     * We check before hashing to avoid unnecessary bcrypt work.
     */
    const existing = await this.prisma.user.findUnique({
      where: { email: dto.email },
    });

    if (existing) {
      throw new ConflictException('Email already registered');
    }

    /*
     * Hash the password using bcrypt.
     * The cost factor is read from environment variables.
     */
    const passwordHash = await bcrypt.hash(dto.password, this.bcryptRounds);

    /*
     * Create the user record in the database.
     */
    const user = await this.prisma.user.create({
      data: {
        firstName: dto.firstName,
        lastName: dto.lastName,
        email: dto.email,
        passwordHash,
        role: 'CLIENT' as any,
        company: dto.company,
        phone: dto.phone,
        wilaya: dto.wilaya,
      },
      select: {
        id: true,
        email: true,
        role: true,
        firstName: true,
        lastName: true,
      },
    });

    /*
     * Generate tokens and create a Redis session.
     */
    const tokens = await this.generateTokensAndSession(
      user.id,
      user.email,
      user.role,
      ip,
      userAgent,
    );

    /*
     * Record the registration event in the audit log.
     */
    await this.audit.log({
      userId: user.id,
      event: AuditEvent.REGISTER,
      ip,
      userAgent,
      metadata: { email: user.email, role: user.role },
    });

    return { user, ...tokens };
  }

  // ----------------------------------------------------------------
  // Login
  // ----------------------------------------------------------------

  /**
   * Authenticate a user and return tokens.
   * Checks account lockout, validates password, handles failed attempts,
   * and creates a new Redis session on success.
   * @param dto       - Login credentials
   * @param ip        - Client IP address
   * @param userAgent - Client user agent
   */
  async login(dto: LoginDto, ip: string, userAgent: string) {
    /*
     * Find the user by email.
     * We select passwordHash explicitly since it is excluded by default.
     */
    const user = await this.prisma.user.findUnique({
      where: { email: dto.email },
      select: {
        id: true,
        email: true,
        role: true,
        firstName: true,
        lastName: true,
        passwordHash: true,
      },
    });

    if (!user) {
      /*
       * Do not reveal whether the email exists.
       * Always return the same error message for invalid credentials.
       */
      await this.audit.log({
        event: AuditEvent.LOGIN_FAILED,
        ip,
        userAgent,
        metadata: { email: dto.email, reason: 'User not found' },
      });
      throw new UnauthorizedException('Invalid credentials');
    }

    /*
     * Check if account is currently locked due to too many failed attempts.
     */
    const lockoutKey = `${REDIS_PREFIX.LOCKOUT}:${user.id}`;
    const isLocked = await this.redis.exists(lockoutKey);

    if (isLocked) {
      const ttl = await this.redis.ttl(lockoutKey);
      const minutesRemaining = Math.ceil(ttl / 60);

      await this.audit.log({
        userId: user.id,
        event: AuditEvent.LOGIN_FAILED,
        ip,
        userAgent,
        metadata: { reason: 'Account locked', minutesRemaining },
      });

      throw new UnauthorizedException(
        `Account is locked. Try again in ${minutesRemaining} minutes`,
      );
    }

    /*
     * Validate the password against the stored hash.
     */
    const passwordValid = await bcrypt.compare(dto.password, user.passwordHash);

    if (!passwordValid) {
      await this.handleFailedAttempt(user.id, ip, userAgent);
      throw new UnauthorizedException('Invalid credentials');
    }

    /*
     * Password is correct. Reset any failed attempt counter.
     */
    await this.redis.del(`${REDIS_PREFIX.ATTEMPTS}:${user.id}`);

    /*
     * Generate tokens and create a Redis session.
     */
    const { passwordHash: _, ...safeUser } = user;
    const tokens = await this.generateTokensAndSession(
      user.id,
      user.email,
      user.role,
      ip,
      userAgent,
    );

    await this.audit.log({
      userId: user.id,
      event: AuditEvent.LOGIN_SUCCESS,
      ip,
      userAgent,
      metadata: { email: user.email },
    });

    return { user: safeUser, ...tokens };
  }

  // ----------------------------------------------------------------
  // Refresh Token
  // ----------------------------------------------------------------

  /**
   * Rotate the refresh token.
   * Validates the old refresh token, deletes it from Redis,
   * and issues a completely new token pair.
   * If reuse of an already-used token is detected, all sessions
   * for that user are immediately revoked.
   * @param refreshToken - The refresh token sent by the client
   * @param ip           - Client IP address
   * @param userAgent    - Client user agent
   */
  async refresh(refreshToken: string, ip: string, userAgent: string) {
    /*
     * Verify the refresh token signature and expiration.
     */
    let payload: any;

    try {
      payload = this.jwt.verify(refreshToken, {
        secret: this.config.get('JWT_REFRESH_SECRET'),
      });
    } catch {
      throw new UnauthorizedException('Invalid or expired refresh token');
    }

    /*
     * Check if this refresh token exists in Redis.
     * If it does not exist, it was already used or never stored.
     * This indicates a token reuse attack.
     */
    const sessionKey = `${REDIS_PREFIX.SESSION}:${payload.sub}:${payload.jti}`;
    const sessionData = await this.redis.getSession(sessionKey);

    if (!sessionData) {
      /*
       * Reuse attack detected.
       * Revoke all sessions for this user immediately.
       */
      this.logger.warn(
        `Refresh token reuse detected for user ${payload.sub} from IP ${ip}`,
      );

      await this.redis.delPattern(`${REDIS_PREFIX.SESSION}:${payload.sub}:*`);

      await this.audit.log({
        userId: payload.sub,
        event: AuditEvent.LOGOUT_ALL,
        ip,
        userAgent,
        metadata: { reason: 'Refresh token reuse attack detected' },
      });

      throw new UnauthorizedException(
        'Security alert: session invalidated. Please log in again',
      );
    }

    /*
     * Token is valid and exists in Redis.
     * Delete the old session immediately before issuing a new one.
     */
    await this.redis.del(sessionKey);

    /*
     * Issue a new token pair and create a new session.
     */
    const tokens = await this.generateTokensAndSession(
      payload.sub,
      payload.email,
      payload.role,
      ip,
      userAgent,
    );

    await this.audit.log({
      userId: payload.sub,
      event: AuditEvent.TOKEN_REFRESHED,
      ip,
      userAgent,
    });

    return tokens;
  }

  // ----------------------------------------------------------------
  // Logout
  // ----------------------------------------------------------------

  /**
   * Log out the current session.
   * Blacklists the access token so it cannot be reused.
   * Deletes the refresh token session from Redis.
   * @param userId        - Current user id
   * @param accessToken   - The access token to blacklist
   * @param refreshToken  - The refresh token to revoke
   * @param jti           - JWT ID of the access token
   * @param ip            - Client IP address
   * @param userAgent     - Client user agent
   */
  async logout(
    userId: string,
    accessToken: string,
    refreshToken: string,
    jti: string,
    ip: string,
    userAgent: string,
  ) {
    /*
     * Blacklist the access token using its jti.
     * TTL is set to ACCESS token lifetime so the key auto-expires.
     */
    const blacklistKey = `${REDIS_PREFIX.BLACKLIST}:${jti}`;
    await this.redis.set(blacklistKey, '1', TOKEN_TTL.ACCESS);

    /*
     * Find and delete the refresh token session from Redis.
     * We verify the refresh token to get its jti, then delete that session.
     */
    try {
      const refreshPayload = this.jwt.verify(refreshToken, {
        secret: this.config.get('JWT_REFRESH_SECRET'),
      }) as any;

      const sessionKey = `${REDIS_PREFIX.SESSION}:${userId}:${refreshPayload.jti}`;
      await this.redis.del(sessionKey);
    } catch {
      /*
       * If refresh token is already expired or invalid, that is fine.
       * The access token blacklist is what matters most here.
       */
      this.logger.warn(
        `Could not revoke refresh token session for user ${userId} during logout`,
      );
    }

    await this.audit.log({
      userId,
      event: AuditEvent.LOGOUT,
      ip,
      userAgent,
    });

    return { message: 'Logged out successfully' };
  }

  // ----------------------------------------------------------------
  // Logout All Sessions
  // ----------------------------------------------------------------

  /**
   * Log out from all devices.
   * Deletes every active session for this user from Redis.
   * The current access token is also blacklisted.
   * @param userId - Current user id
   * @param jti    - JWT ID of the current access token to blacklist
   * @param ip     - Client IP address
   * @param userAgent - Client user agent
   */
  async logoutAll(
    userId: string,
    jti: string,
    ip: string,
    userAgent: string,
  ) {
    /*
     * Blacklist the current access token.
     */
    const blacklistKey = `${REDIS_PREFIX.BLACKLIST}:${jti}`;
    await this.redis.set(blacklistKey, '1', TOKEN_TTL.ACCESS);

    /*
     * Delete all sessions for this user from Redis.
     */
    await this.redis.delPattern(`${REDIS_PREFIX.SESSION}:${userId}:*`);

    await this.audit.log({
      userId,
      event: AuditEvent.LOGOUT_ALL,
      ip,
      userAgent,
      metadata: { reason: 'User requested logout from all devices' },
    });

    return { message: 'Logged out from all devices successfully' };
  }

  // ----------------------------------------------------------------
  // Change Password
  // ----------------------------------------------------------------

  /**
   * Change password while authenticated.
   * Validates current password, updates the hash,
   * and revokes all other sessions to force re-login on other devices.
   * @param userId  - Current user id
   * @param jti     - JWT ID of the current access token
   * @param dto     - Current and new password
   * @param ip      - Client IP address
   * @param userAgent - Client user agent
   */
  async changePassword(
    userId: string,
    jti: string,
    dto: ChangePasswordDto,
    ip: string,
    userAgent: string,
  ) {
    /*
     * Load user with password hash for comparison.
     */
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: { id: true, passwordHash: true },
    });

    if (!user) {
      throw new UnauthorizedException('User not found');
    }

    /*
     * Verify the current password is correct.
     */
    const passwordValid = await bcrypt.compare(
      dto.currentPassword,
      user.passwordHash,
    );

    if (!passwordValid) {
      throw new BadRequestException('Current password is incorrect');
    }

    /*
     * Hash and save the new password.
     */
    const newHash = await bcrypt.hash(dto.newPassword, this.bcryptRounds);

    await this.prisma.user.update({
      where: { id: userId },
      data: { passwordHash: newHash },
    });

    /*
     * Revoke all sessions on other devices.
     * The current session access token is blacklisted too,
     * forcing a fresh login everywhere.
     */
    const blacklistKey = `${REDIS_PREFIX.BLACKLIST}:${jti}`;
    await this.redis.set(blacklistKey, '1', TOKEN_TTL.ACCESS);
    await this.redis.delPattern(`${REDIS_PREFIX.SESSION}:${userId}:*`);

    await this.audit.log({
      userId,
      event: AuditEvent.PASSWORD_CHANGED,
      ip,
      userAgent,
    });

    return { message: 'Password changed successfully. Please log in again' };
  }

  // ----------------------------------------------------------------
  // Forgot Password
  // ----------------------------------------------------------------

  /**
   * Generate a password reset token and store it in Redis.
   * The token is returned directly in the response (no email).
   * In production, this token would be sent via email instead.
   * @param dto - Email address of the account
   * @param ip  - Client IP address
   */
  async forgotPassword(dto: ForgotPasswordDto, ip: string) {
    const user = await this.prisma.user.findUnique({
      where: { email: dto.email },
      select: { id: true, email: true },
    });

    /*
     * Always return success even if email is not found.
     * This prevents email enumeration attacks.
     */
    if (!user) {
      return {
        message: 'If this email exists, a reset token has been generated',
      };
    }

    /*
     * Generate a unique reset token and store it in Redis
     * with a 15-minute TTL. Any previous reset token is overwritten.
     */
    const resetToken = uuidv4();
    const resetKey = `${REDIS_PREFIX.RESET}:${resetToken}`;
    await this.redis.set(resetKey, user.id, TOKEN_TTL.RESET);

    await this.audit.log({
      userId: user.id,
      event: AuditEvent.PASSWORD_RESET_REQUESTED,
      ip,
      metadata: { email: user.email },
    });

    /*
     * In development, return the token directly.
     * In production, send this token via email and return only the message.
     */
    return {
      message: 'If this email exists, a reset token has been generated',
      resetToken,  // remove this in production
    };
  }

  // ----------------------------------------------------------------
  // Reset Password
  // ----------------------------------------------------------------

  /**
   * Reset the password using a valid reset token.
   * The token is deleted from Redis after use so it cannot be reused.
   * @param dto - Reset token and new password
   * @param ip  - Client IP address
   */
  async resetPassword(dto: ResetPasswordDto, ip: string) {
    /*
     * Look up the reset token in Redis to get the associated user id.
     */
    const resetKey = `${REDIS_PREFIX.RESET}:${dto.token}`;
    const userId = await this.redis.get(resetKey);

    if (!userId) {
      throw new BadRequestException('Reset token is invalid or has expired');
    }

    /*
     * Hash and save the new password.
     */
    const newHash = await bcrypt.hash(dto.newPassword, this.bcryptRounds);

    await this.prisma.user.update({
      where: { id: userId },
      data: { passwordHash: newHash },
    });

    /*
     * Delete the reset token immediately so it cannot be used again.
     */
    await this.redis.del(resetKey);

    /*
     * Revoke all active sessions to force re-login.
     */
    await this.redis.delPattern(`${REDIS_PREFIX.SESSION}:${userId}:*`);

    await this.audit.log({
      userId,
      event: AuditEvent.PASSWORD_RESET_SUCCESS,
      ip,
      metadata: { tokenUsed: dto.token },
    });

    return { message: 'Password reset successfully. Please log in again' };
  }

  // ----------------------------------------------------------------
  // Private Helpers
  // ----------------------------------------------------------------

  /**
   * Generate an access token and a refresh token,
   * then store a session in Redis for the refresh token.
   * Both tokens contain a unique jti for blacklisting and rotation.
   * @param userId    - User id to embed in the token
   * @param email     - User email to embed in the token
   * @param role      - User role to embed in the token
   * @param ip        - Client IP address to store in session
   * @param userAgent - Client user agent to store in session
   */
  private async generateTokensAndSession(
    userId: string,
    email: string,
    role: string,
    ip: string,
    userAgent: string,
  ) {
    /*
     * Each token gets a unique jti so they can be individually revoked.
     */
    const accessJti = uuidv4();
    const refreshJti = uuidv4();

    const accessToken = this.jwt.sign(
      { sub: userId, email, role, jti: accessJti },
      {
        secret: this.config.get('JWT_ACCESS_SECRET'),
        expiresIn: this.config.get('JWT_ACCESS_EXPIRATION') || '15m',
      },
    );

    const refreshToken = this.jwt.sign(
      { sub: userId, email, role, jti: refreshJti },
      {
        secret: this.config.get('JWT_REFRESH_SECRET'),
        expiresIn: this.config.get('JWT_REFRESH_EXPIRATION') || '7d',
      },
    );

    /*
     * Store the refresh token session in Redis.
     * Key format: session:{userId}:{refreshJti}
     * This allows deleting all sessions for a user with a pattern delete.
     */
    const sessionKey = `${REDIS_PREFIX.SESSION}:${userId}:${refreshJti}`;

    await this.redis.setSession(
      sessionKey,
      {
        userId,
        email,
        role,
        ip,
        userAgent,
        createdAt: new Date().toISOString(),
      },
      TOKEN_TTL.REFRESH,
    );

    return { accessToken, refreshToken };
  }

  /**
   * Handle a failed login attempt.
   * Increments the attempt counter in Redis.
   * Locks the account if the maximum attempts are exceeded.
   * @param userId    - User id of the failed login
   * @param ip        - Client IP address
   * @param userAgent - Client user agent
   */
  private async handleFailedAttempt(
    userId: string,
    ip: string,
    userAgent: string,
  ) {
    const attemptsKey = `${REDIS_PREFIX.ATTEMPTS}:${userId}`;

    /*
     * Get current attempt count and increment it.
     */
    const current = await this.redis.get(attemptsKey);
    const attempts = current ? parseInt(current) + 1 : 1;

    /*
     * Store the updated count with a TTL matching the lockout window.
     */
    await this.redis.set(
      attemptsKey,
      attempts.toString(),
      this.lockoutDuration,
    );

    await this.audit.log({
      userId,
      event: AuditEvent.LOGIN_FAILED,
      ip,
      userAgent,
      metadata: { attempts },
    });

    /*
     * Lock the account if max attempts are exceeded.
     */
    if (attempts >= this.maxAttempts) {
      const lockoutKey = `${REDIS_PREFIX.LOCKOUT}:${userId}`;
      await this.redis.set(lockoutKey, '1', this.lockoutDuration);

      await this.audit.log({
        userId,
        event: AuditEvent.ACCOUNT_LOCKED,
        ip,
        userAgent,
        metadata: {
          attempts,
          lockoutDurationMinutes: this.lockoutDuration / 60,
        },
      });

      this.logger.warn(
        `Account ${userId} locked after ${attempts} failed attempts from IP ${ip}`,
      );
    }
  }
}