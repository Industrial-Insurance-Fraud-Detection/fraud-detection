import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { EquipmentModule } from './equipment/equipment.module';
import { ClaimsModule } from './claims/claims.module';
import { FilesModule } from './files/files.module';
import { QueueModule } from './queue/queue.module';
import { NotificationsModule } from './notifications/notifications.module';
import { PrismaModule } from './prisma/prisma.module';
import { CommonModule } from './common/common.module';
import { HealthModule } from './health/health.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    PrismaModule,
    CommonModule,
    AuthModule,
    UsersModule,
    EquipmentModule,
    ClaimsModule,
    FilesModule,
    QueueModule,
    NotificationsModule,
    HealthModule,
  ],
})
export class AppModule { }