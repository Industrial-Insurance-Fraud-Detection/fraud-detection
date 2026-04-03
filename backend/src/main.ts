
import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import helmet from 'helmet';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';
import { TransformInterceptor } from './common/interceptors/transform.interceptor';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  /*
   * Helmet
   * Sets secure HTTP response headers to protect against common
   * web vulnerabilities such as XSS, clickjacking, and MIME sniffing.
   * Must be applied before any routes are registered.
   */
  app.use(helmet());

  /*
   * CORS
   * Only allows requests from the configured frontend origin.
   * In production, FRONTEND_URL must be set to the real domain.
   */
  app.enableCors({
    origin: process.env.FRONTEND_URL || 'http://localhost:5173',
    methods: ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
  });

  /*
   * Global prefix
   * All routes will be prefixed with /api/v1
   */
  app.setGlobalPrefix('api/v1');

  /*
   * Global validation pipe
   * Validates all incoming request bodies against their DTOs.
   * Strips unknown properties and transforms types automatically.
   */
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidNonWhitelisted: false, // ✅ allow extra fields like "files"
      transformOptions: {
        enableImplicitConversion: true,
      },
    }),
  );
  /*
   * Global exception filter
   * Returns a uniform error shape for all thrown exceptions.
   */
  app.useGlobalFilters(new HttpExceptionFilter());

  /*
   * Global interceptor
   * Wraps all successful responses in { success: true, data: ... }
   */
  app.useGlobalInterceptors(new TransformInterceptor());

  /*
   * Swagger documentation
   * Available at /api/docs in development.
   */
  const config = new DocumentBuilder()
    .setTitle('Taamine API')
    .setDescription('AI-Powered Industrial Insurance Fraud Detection')
    .setVersion('1.0')
    .addBearerAuth()
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  const port = process.env.PORT || 3000;
  await app.listen(port);

  console.log(`Taamine backend running on http://localhost:${port}`);
  console.log(`Swagger docs at http://localhost:${port}/api/docs`);
}

bootstrap();