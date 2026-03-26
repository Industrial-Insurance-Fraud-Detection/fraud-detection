import { PrismaClient, Role } from '@prisma/client';
import * as bcrypt from 'bcryptjs';
import * as dotenv from 'dotenv';
dotenv.config();
const prisma = new PrismaClient();

async function main() {
  console.log(' Seeding database...');

  // Create demo investigator
  const investigator = await prisma.user.upsert({
    where: { email: 'investigator@taamine.dz' },
    update: {},
    create: {
      email: 'investigator@taamine.dz',
      passwordHash: await bcrypt.hash('password123', 10),
      role: Role.INVESTIGATOR,
      firstName: 'Karim',
      lastName: 'Meziani',
      company: 'CAAT Insurance',
      phone: '0770000001',
      wilaya: 'Alger',
    },
  });

  // Create demo client
  const client = await prisma.user.upsert({
    where: { email: 'client@sonatrach.dz' },
    update: {},
    create: {
      email: 'client@sonatrach.dz',
      passwordHash: await bcrypt.hash('password123', 10),
      role: Role.CLIENT,
      firstName: 'Ahmed',
      lastName: 'Benali',
      company: 'Sonatrach',
      phone: '0550000002',
      wilaya: 'Boumerdès',
    },
  });

  // Create demo equipment
  await prisma.equipment.upsert({
    where: { serialNumber: 'AC-GA55-2019-001' },
    update: {},
    create: {
      ownerId: client.id,
      name: 'Compresseur Atlas Copco GA-55',
      type: 'Compresseur',
      manufacturer: 'Atlas Copco',
      model: 'GA-55',
      serialNumber: 'AC-GA55-2019-001',
      commissionDate: new Date('2019-06-15'),
      location: 'Usine Boumerdès — Bâtiment B2',
    },
  });

  console.log('✅ Seed complete');
  console.log('   Investigator → investigator@taamine.dz / password123');
  console.log('   Client       → client@sonatrach.dz    / password123');
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
