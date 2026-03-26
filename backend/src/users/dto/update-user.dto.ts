import {
  IsOptional,
  IsString,
  IsPhoneNumber,
  MinLength,
  MaxLength,
  Matches,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * UpdateUserDto
 * Validates profile update input.
 * All fields are optional — client sends only what they want to change.
 * Role and email cannot be changed via this endpoint.
 */
export class UpdateUserDto {
  @ApiProperty({ required: false, example: 'Ahmed' })
  @IsOptional()
  @IsString()
  @MinLength(2)
  @MaxLength(50)
  @Matches(/^[a-zA-ZÀ-ÿ\s'-]+$/, {
    message: 'firstName must contain letters only',
  })
  firstName?: string;

  @ApiProperty({ required: false, example: 'Benali' })
  @IsOptional()
  @IsString()
  @MinLength(2)
  @MaxLength(50)
  @Matches(/^[a-zA-ZÀ-ÿ\s'-]+$/, {
    message: 'lastName must contain letters only',
  })
  lastName?: string;

  @ApiProperty({ required: false, example: '+213555123456' })
  @IsOptional()
  @IsPhoneNumber('DZ', {
    message: 'phone must be a valid Algerian phone number',
  })
  phone?: string;

  @ApiProperty({ required: false, example: 'Boumerdès' })
  @IsOptional()
  @IsString()
  @MinLength(2)
  @MaxLength(100)
  wilaya?: string;

  @ApiProperty({ required: false, example: 'Boumerdès centre' })
  @IsOptional()
  @IsString()
  @MinLength(2)
  @MaxLength(100)
  commune?: string;

  @ApiProperty({ required: false, example: 'Sonatrach, Département Maintenance' })
  @IsOptional()
  @IsString()
  @MinLength(2)
  @MaxLength(200)
  company?: string;
}