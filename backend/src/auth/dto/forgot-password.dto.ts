import { IsEmail } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * ForgotPasswordDto
 * Validates the request body for POST /auth/forgot-password.
 */
export class ForgotPasswordDto {
    @ApiProperty({ example: 'ahmed@sonatrach.dz' })
    @IsEmail()
    email: string;
}