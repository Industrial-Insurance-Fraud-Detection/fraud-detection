/*
  Warnings:

  - You are about to drop the column `companyName` on the `users` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "users" DROP COLUMN "companyName",
ADD COLUMN     "commune" TEXT,
ADD COLUMN     "company" TEXT;
