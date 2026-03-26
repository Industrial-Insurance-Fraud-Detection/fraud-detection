import { Injectable, UnauthorizedException } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../../prisma/prisma.service';
import { RedisService } from '../../common/services/redis.service';

/**
 * JWT payload structure.
 * This is what gets encoded inside every access token.
 */
export interface JwtPayload {
  sub: string;
  email: string;
  role: string;
  jti: string;
}

/**
 * JwtStrategy
 * Validates the access token on every protected request.
 * Two checks are performed:
 * 1. Token signature and expiration (handled by passport-jwt)
 * 2. Token blacklist check in Redis (handles logout invalidation)
 * If either check fails the request is rejected with 401.
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
      passReqToCallback: true,
    });
  }

  /**
   * Called automatically by passport after token signature is verified.
   * Checks if the token has been blacklisted (user logged out).
   * Checks if the user still exists in the database.
   * Attaches the user object to req.user if everything passes.
   * @param req     - The incoming HTTP request
   * @param payload - The decoded JWT payload
   */
  async validate(req: any, payload: JwtPayload) {
    /*
     * Check if this token's jti is in the Redis blacklist.
     * Tokens are blacklisted on logout and remain there
     * until their natural expiration time.
     */
    const isBlacklisted = await this.redis.exists(
      `blacklist:${payload.jti}`,
    );

    if (isBlacklisted) {
      throw new UnauthorizedException('Token has been revoked');
    }

    /*
     * Verify the user still exists in the database.
     * Handles the case where a user was deleted after the token was issued.
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
     * Return value is attached to req.user automatically by passport.
     * We include jti so the auth service can blacklist
     * this specific token on logout.
     */
    return {
      id: user.id,
      email: user.email,
      role: user.role,
      firstName: user.firstName,
      lastName: user.lastName,
      jti: payload.jti,
    };
  }
}