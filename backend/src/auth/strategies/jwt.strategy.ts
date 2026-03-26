import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../../prisma/prisma.service';
import { RedisService } from '../../common/services/redis.service';

/**
 * JWT payload structure.
 * Every access token contains these fields.
 */
export interface JwtPayload {
  sub: string;  // user id
  email: string;
  role: string;
  jti: string;  // unique token id, used for blacklisting
}

/**
 * JwtStrategy
 * Validates every incoming access token on protected routes.
 * Two checks are performed beyond signature verification:
 * 1. The token jti must not be in the Redis blacklist (logged out tokens)
 * 2. The user must still exist in the database
 */
@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy, 'jwt') {
  constructor(
    config: ConfigService,
    private prisma: PrismaService,
    private redis: RedisService,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      secretOrKey: config.get<string>('JWT_ACCESS_SECRET'),
      ignoreExpiration: false,
    });
  }

  /**
   * Called automatically by Passport after signature verification.
   * Checks blacklist then loads the user from the database.
   * Returning the user object attaches it to req.user.
   * @param payload - Decoded JWT payload
   */
  async validate(payload: JwtPayload) {
    /*
     * Check if this token has been blacklisted.
     * Tokens are blacklisted on logout with a TTL equal
     * to their remaining lifetime, so this check is always accurate.
     */
    const blacklistKey = `blacklist:${payload.jti}`;
    const isBlacklisted = await this.redis.exists(blacklistKey);

    if (isBlacklisted) {
      throw new UnauthorizedException('Token has been revoked');
    }

    /*
     * Verify the user still exists in the database.
     * This catches cases where a user was deleted after the token was issued.
     */
    const user = await this.prisma.user.findUnique({
      where: { id: payload.sub },
      select: {
        id: true,
        email: true,
        role: true,
        firstName: true,
        lastName: true,
      },
    });

    if (!user) {
      throw new UnauthorizedException('User no longer exists');
    }

    /*
     * Return the user object.
     * This gets attached to req.user and is available
     * in all controllers via @CurrentUser() decorator.
     */
    return { ...user, jti: payload.jti };
  }
}