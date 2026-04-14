FROM node:20-alpine
RUN apk add --no-cache openssl openssl-dev
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npx prisma generate
RUN npm run build
EXPOSE 3000
CMD ["node", "dist/src/main"]