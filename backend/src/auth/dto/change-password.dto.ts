import { IsString, IsNotEmpty, MinLength, MaxLength, Matches } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * ChangePasswordDto
 * Validates the request body for POST /auth/change-password.
 */
export class ChangePasswordDto {
    @ApiProperty({ example: 'OldPass123' })
    @IsString()
    @IsNotEmpty()
    currentPassword: string;

    @ApiProperty({ example: 'NewPass456', minLength: 8 })
    @IsString()
    @MinLength(8)
    @MaxLength(64)
    @Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/, {
        message: 'Password must contain at least one uppercase letter, one lowercase letter, and one number',
    })
    newPassword: string;
}