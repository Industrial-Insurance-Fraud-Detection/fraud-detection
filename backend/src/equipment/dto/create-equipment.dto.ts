import {
  IsDateString,
  IsNotEmpty,
  IsOptional,
  IsString,
  MaxLength,
  MinLength,
  Matches,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * CreateEquipmentDto
 * Validates input when registering a new industrial machine.
 * commissionDate must be in the past — cannot register a machine not yet in service.
 */
export class CreateEquipmentDto {
  @ApiProperty({ example: 'Compresseur Atlas Copco GA-55' })
  @IsString()
  @IsNotEmpty()
  @MinLength(3)
  @MaxLength(100)
  name: string;

  @ApiProperty({
    example: 'Compresseur',
    description: 'Must be one of the supported equipment types',
    enum: [
      'Pompe Industrielle',
      'Compresseur',
      'Moteur Electrique',
      'Generateur',
      'Turbine',
      'Pompe Hydraulique',
    ],
  })
  @IsString()
  @IsNotEmpty()
  type: string;

  @ApiProperty({ example: 'Atlas Copco', required: false })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  manufacturer?: string;

  @ApiProperty({ example: 'GA-55', required: false })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  model?: string;

  @ApiProperty({ example: 'AC-GA55-2019-001' })
  @IsString()
  @IsNotEmpty()
  @Matches(/^[A-Z0-9\-]+$/, {
    message: 'serialNumber must contain uppercase letters, numbers, and hyphens only',
  })
  @MaxLength(50)
  serialNumber: string;

  @ApiProperty({
    example: '2019-06-15',
    description: 'Commission date must be in the past',
  })
  @IsDateString()
  commissionDate: string;

  @ApiProperty({ example: 'Usine Boumerdès — Bâtiment B2', required: false })
  @IsOptional()
  @IsString()
  @MaxLength(200)
  location?: string;
}