import {
  IsDateString,
  IsNotEmpty,
  IsNumber,
  IsPositive,
  IsString,
  MaxLength,
  MinLength,
  Max,
} from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';
import { Type } from 'class-transformer';

/**
 * CreateClaimDto
 * Validates input when submitting a new insurance claim.
 * Files are handled separately via multipart/form-data.
 * incidentDate cannot be in the future.
 */
export class CreateClaimDto {
  @ApiProperty({ example: 'clx123abc456' })
  @IsString()
  @IsNotEmpty()
  equipmentId: string;

  @ApiProperty({
    example: '2026-02-28',
    description: 'Date of the incident — cannot be in the future',
  })
  @IsDateString()
  incidentDate: string;

  @ApiProperty({
    example: 'La pompe hydraulique a subi une surchauffe soudaine suite à un défaut de refroidissement.',
    description: 'Detailed description of the incident — minimum 20 characters',
  })
  @IsString()
  @IsNotEmpty()
  @MinLength(20, { message: 'description must be at least 20 characters' })
  @MaxLength(2000, { message: 'description cannot exceed 2000 characters' })
  description: string;

  @ApiProperty({
    example: 450000,
    description: 'Claimed amount in DZD — must be positive and under 500,000,000',
  })
  @Type(() => Number)
  @IsNumber()
  @IsPositive()
  @Max(500_000_000, { message: 'claimedAmount cannot exceed 500,000,000 DZD' })
  claimedAmount: number;
}