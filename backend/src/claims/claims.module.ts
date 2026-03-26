import { Module } from '@nestjs/common';
import { ClaimsController } from './claims.controller';
import { ClaimsService } from './claims.service';
import { FilesModule } from '../files/files.module';
import { QueueModule } from '../queue/queue.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { EquipmentModule } from '../equipment/equipment.module';

/**
 * ClaimsModule
 * Imports EquipmentModule to use verifyActiveAndOwned
 * before allowing a claim submission.
 */
@Module({
  imports: [FilesModule, QueueModule, NotificationsModule, EquipmentModule],
  controllers: [ClaimsController],
  providers: [ClaimsService],
})
export class ClaimsModule { }