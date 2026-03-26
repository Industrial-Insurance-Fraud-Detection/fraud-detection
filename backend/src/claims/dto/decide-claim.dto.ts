import { IsEnum, IsNotEmpty, IsString, MinLength, MaxLength } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';
import { DecisionOutcome } from '@prisma/client';

/**
 * DecideClaimDto
 * Validates the investigator's decision on a claim.
 * Notes are mandatory — investigator must justify every decision.
 */
export class DecideClaimDto {
  @ApiProperty({
    enum: DecisionOutcome,
    example: DecisionOutcome.APPROVED,
    description: 'Final decision on the claim',
  })
  @IsEnum(DecisionOutcome)
  outcome: DecisionOutcome;

  @ApiProperty({
    example: 'Analyse des capteurs confirme une vraie panne thermique. Aucun signe de manipulation.',
    description: 'Mandatory justification for the decision — minimum 10 characters',
  })
  @IsString()
  @IsNotEmpty()
  @MinLength(10, { message: 'notes must be at least 10 characters' })
  @MaxLength(2000, { message: 'notes cannot exceed 2000 characters' })
  notes: string;
}