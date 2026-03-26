import { SetMetadata } from '@nestjs/common';
import { Role } from '@prisma/client';

export const ROLES_KEY = 'roles';

/** Usage: @Roles(Role.CLIENT) or @Roles(Role.INVESTIGATOR) */
export const Roles = (...roles: Role[]) => SetMetadata(ROLES_KEY, roles);
