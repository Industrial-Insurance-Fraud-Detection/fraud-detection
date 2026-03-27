import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { JwtStrategy } from './strategies/jwt.strategy';

/**
 * AuthModule
 * Registers all auth-related providers, controllers, and strategies.
 *
 * RedisService and AuditService are injected automatically from CommonModule
 * since CommonModule is marked as @Global() in app.module.ts.
 *
 * JwtModule is registered without a default secret because each token type
 * uses its own secret, passed dynamically in auth.service.ts via jwt.sign()
 * and jwt.verify() with explicit options.
 *
 * JwtStrategy is registered here so Passport can resolve the 'jwt' guard
 * across the entire application. It validates every incoming access token
 * against the Redis blacklist and confirms the user still exists.
 */
@Module({
  imports: [
    PassportModule,
    JwtModule.register({}),
  ],
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy],
  exports: [AuthService],
})
export class AuthModule { }