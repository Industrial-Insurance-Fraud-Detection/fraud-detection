import {
  Controller,
  Post,
  Body,
  UseGuards,
  HttpCode,
  HttpStatus,
  Req,
} from '@nestjs/common';
import { ApiTags, ApiOperation, ApiBearerAuth } from '@nestjs/swagger';
import { Request } from 'express';
import { AuthService } from './auth.service';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { RefreshTokenDto } from './dto/refresh-token.dto';
import { ChangePasswordDto } from './dto/change-password.dto';
import { ForgotPasswordDto } from './dto/forgot-password.dto';
import { ResetPasswordDto } from './dto/reset-password.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';

/**
 * AuthController
 * Handles all authentication endpoints.
 * Public routes require no guard.
 * Protected routes require a valid access token via JwtAuthGuard.
 */
@ApiTags('Auth')
@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) { }

  // ----------------------------------------------------------------
  // Public Routes
  // ----------------------------------------------------------------

  @Post('register')
  @ApiOperation({ summary: 'Create a new account' })
  register(@Body() dto: RegisterDto, @Req() req: Request) {
    return this.authService.register(
      dto,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  @Post('login')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Login and receive access and refresh tokens' })
  login(@Body() dto: LoginDto, @Req() req: Request) {
    return this.authService.login(
      dto,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  @Post('refresh')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Rotate refresh token and get new token pair' })
  refresh(@Body() dto: RefreshTokenDto, @Req() req: Request) {
    return this.authService.refresh(
      dto.refreshToken,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  @Post('forgot-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Request a password reset token' })
  forgotPassword(@Body() dto: ForgotPasswordDto, @Req() req: Request) {
    return this.authService.forgotPassword(dto, this.getIp(req));
  }

  @Post('reset-password')
  @HttpCode(HttpStatus.OK)
  @ApiOperation({ summary: 'Reset password using a valid reset token' })
  resetPassword(@Body() dto: ResetPasswordDto, @Req() req: Request) {
    return this.authService.resetPassword(dto, this.getIp(req));
  }

  // ----------------------------------------------------------------
  // Protected Routes
  // ----------------------------------------------------------------

  @Post('logout')
  @HttpCode(HttpStatus.OK)
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Logout current session' })
  logout(
    @CurrentUser() user: any,
    @Body() body: RefreshTokenDto,
    @Req() req: Request,
  ) {
    /*
     * Extract the raw access token from the Authorization header.
     * Format: "Bearer <token>"
     */
    const accessToken = req.headers.authorization?.split(' ')[1];

    return this.authService.logout(
      user.id,
      accessToken,
      body.refreshToken,
      user.jti,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  @Post('logout-all')
  @HttpCode(HttpStatus.OK)
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Logout from all devices' })
  logoutAll(@CurrentUser() user: any, @Req() req: Request) {
    return this.authService.logoutAll(
      user.id,
      user.jti,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  @Post('change-password')
  @HttpCode(HttpStatus.OK)
  @UseGuards(JwtAuthGuard)
  @ApiBearerAuth()
  @ApiOperation({ summary: 'Change password while authenticated' })
  changePassword(
    @CurrentUser() user: any,
    @Body() dto: ChangePasswordDto,
    @Req() req: Request,
  ) {
    return this.authService.changePassword(
      user.id,
      user.jti,
      dto,
      this.getIp(req),
      this.getUserAgent(req),
    );
  }

  // ----------------------------------------------------------------
  // Private Helpers
  // ----------------------------------------------------------------

  /**
   * Extract the client IP address from the request.
   * Checks x-forwarded-for header first to handle proxies and load balancers.
   * Falls back to the direct connection IP.
   * @param req - Express request object
   */
  private getIp(req: Request): string {
    return (
      (req.headers['x-forwarded-for'] as string)?.split(',')[0].trim() ||
      req.ip ||
      'unknown'
    );
  }

  /**
   * Extract the user agent string from the request headers.
   * Used for session tracking and audit logging.
   * @param req - Express request object
   */
  private getUserAgent(req: Request): string {
    return req.headers['user-agent'] || 'unknown';
  }
}