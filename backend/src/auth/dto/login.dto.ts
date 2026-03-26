import { IsEmail, IsString, IsNotEmpty } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * LoginDto
 * Validates the request body for POST /auth/login.
 */
export class LoginDto {
  @ApiProperty({ example: 'ahmed@sonatrach.dz' })
  @IsEmail()
  email: string;

  @ApiProperty({ example: 'StrongPass123' })
  @IsString()
  @IsNotEmpty()
  password: string;
}