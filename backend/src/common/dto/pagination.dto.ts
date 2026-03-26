import { IsOptional, IsInt, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

/**
 * PaginationDto
 * Reusable pagination parameters for all list endpoints.
 * Used by equipment, claims, notifications, and audit logs.
 */
export class PaginationDto {
    @ApiProperty({ required: false, default: 1 })
    @IsOptional()
    @Type(() => Number)
    @IsInt()
    @Min(1)
    page?: number = 1;

    @ApiProperty({ required: false, default: 10 })
    @IsOptional()
    @Type(() => Number)
    @IsInt()
    @Min(1)
    @Max(100)
    limit?: number = 10;
}

/**
 * Creates a standardized paginated response object.
 * Used by all list endpoints to return consistent pagination metadata.
 * @param data  - Array of records for the current page
 * @param total - Total number of records across all pages
 * @param page  - Current page number
 * @param limit - Records per page
 */
export function paginate<T>(data: T[], total: number, page: number, limit: number) {
    return {
        data,
        pagination: {
            total,
            page,
            limit,
            totalPages: Math.ceil(total / limit),
            hasNextPage: page < Math.ceil(total / limit),
            hasPrevPage: page > 1,
        },
    };
}