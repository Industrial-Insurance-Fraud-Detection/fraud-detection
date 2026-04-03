import { SetMetadata } from '@nestjs/common';

export const IS_PUBLIC_KEY = 'isPublic';

/**
 * Mark a route as public — skips JwtAuthGuard even when applied at class level.
 * Used for internal service-to-service endpoints called by n8n without a user token.
 */
export const Public = () => SetMetadata(IS_PUBLIC_KEY, true);