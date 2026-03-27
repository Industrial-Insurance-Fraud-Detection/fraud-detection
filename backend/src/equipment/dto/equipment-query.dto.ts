import { IsOptional, IsString, MaxLength } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';
import { PaginationDto } from '../../common/dto/pagination.dto';

/**
 * EquipmentQueryDto
 * Extends PaginationDto with optional search and type filters.
 * Needed because ValidationPipe has forbidNonWhitelisted: true globally —
 * any query param not declared in the DTO is rejected with 400.
 */
export class EquipmentQueryDto extends PaginationDto {
    @ApiProperty({ required: false, example: 'Atlas' })
    @IsOptional()
    @IsString()
    @MaxLength(100)
    search?: string;

    @ApiProperty({ required: false, example: 'Compresseur' })
    @IsOptional()
    @IsString()
    @MaxLength(100)
    type?: string;
}