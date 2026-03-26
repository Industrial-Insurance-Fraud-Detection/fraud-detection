import {
  IsDateString,
  IsOptional,
  IsString,
  IsBoolean,
  MaxLength,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

/**
 * UpdateEquipmentDto
 * All fields optional — client sends only what they want to change.
 * serialNumber and type cannot be changed after registration.
 */
export class UpdateEquipmentDto {
  @ApiProperty({ required: false, example: 'Compresseur Atlas Copco GA-55 v2' })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  name?: string;

  @ApiProperty({ required: false, example: 'Usine Boumerdès — Bâtiment B3' })
  @IsOptional()
  @IsString()
  @MaxLength(200)
  location?: string;

  @ApiProperty({ required: false, example: 'Atlas Copco' })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  manufacturer?: string;

  @ApiProperty({ required: false, example: 'GA-55 Pro' })
  @IsOptional()
  @IsString()
  @MaxLength(100)
  model?: string;

  @ApiProperty({
    required: false,
    example: '2019-06-15',
    description: 'Cannot be set to a future date',
  })
  @IsOptional()
  @IsDateString()
  commissionDate?: string;

  @ApiProperty({ required: false, example: '2024-03-01' })
  @IsOptional()
  @IsDateString()
  lastMaintenanceDate?: string;

  @ApiProperty({ required: false })
  @IsOptional()
  @IsBoolean()
  isActive?: boolean;
}