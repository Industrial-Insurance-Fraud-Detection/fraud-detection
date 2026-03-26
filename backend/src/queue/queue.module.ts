import { Module } from '@nestjs/common';
import { RabbitMQModule } from '@golevelup/nestjs-rabbitmq';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { QueueProducer } from './queue.producer';
import { QueueWorker } from './queue.worker';
import { NotificationsModule } from '../notifications/notifications.module';

@Module({
  imports: [
    RabbitMQModule.forRootAsync(RabbitMQModule, {
      imports: [ConfigModule],
      useFactory: (config: ConfigService) => ({
        uri: config.get<string>('RABBITMQ_URL') || 'amqp://guest:guest@localhost:5672',
        exchanges: [
          {
            name: 'taamine',
            type: 'direct',
          },
        ],
        connectionInitOptions: { wait: false }, // don't block startup if RabbitMQ not ready
      }),
      inject: [ConfigService],
    }),
    NotificationsModule,
  ],
  providers: [QueueProducer, QueueWorker],
  exports: [QueueProducer],
})
export class QueueModule {}
