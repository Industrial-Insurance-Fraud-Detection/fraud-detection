import {
  IsEmail,
  IsNotEmpty,
  IsOptional,
  IsString,
  MinLength,
  MaxLength,
  Matches,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';


/**
 * RegisterDto
 * Validates the request body for POST /auth/register.
 * Password must contain at least one uppercase letter,
 * one lowercase letter, and one number.
 */
export class RegisterDto {
  @ApiProperty({ example: 'Ahmed' })
  @IsString()
  @IsNotEmpty()
  @MaxLength(50)
  firstName: string;

  @ApiProperty({ example: 'Benali' })
  @IsString()
  @IsNotEmpty()
  @MaxLength(50)
  lastName: string;

  @ApiProperty({ example: 'ahmed@sonatrach.dz' })
  @IsEmail()
  email: string;

  @ApiProperty({ example: 'StrongPass123', minLength: 8 })
  @IsString()
  @MinLength(8)
  @MaxLength(64)
  @Matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/, {
    message: 'Password must contain at least one uppercase letter, one lowercase letter, and one number',
  })
  password: string;


  @ApiProperty({ example: 'Sonatrach', required: false })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  company?: string;

  @ApiProperty({ example: '0550000001', required: false })
  @IsOptional()
  @IsString()
  phone?: string;

  @ApiProperty({ example: 'Boumerdes', required: false })
  @IsOptional()
  @IsString()
  wilaya?: string;
}