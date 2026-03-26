import { Injectable, Logger } from '@nestjs/common';
import { AmqpConnection } from '@golevelup/nestjs-rabbitmq';

export interface AnalysisJobPayload {
  claimId: string;
}

@Injectable()
export class QueueProducer {
  private readonly logger = new Logger(QueueProducer.name);

  constructor(private amqp: AmqpConnection) {}

  /** Push an AI analysis job to the RabbitMQ queue */
  async publishAnalysisJob(payload: AnalysisJobPayload): Promise<void> {
    await this.amqp.publish(
      'taamine',          // exchange name
      'ai-analysis',      // routing key
      payload,
    );
    this.logger.log(`📤 Analysis job queued for claim: ${payload.claimId}`);
  }
}
