FROM node:24-slim

WORKDIR /app

# npm ci, not npm install: it installs exactly what package-lock.json
# pins and fails when the lockfile and package.json disagree. Five people
# building this image have to get the same dependency tree.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
