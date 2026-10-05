FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG KADAN_COMMIT=development
RUN npm run build && node -e 'require("fs").writeFileSync("dist/version.json", JSON.stringify({commit: process.argv[1]}))' "$KADAN_COMMIT"

FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
ARG KADAN_COMMIT=development
LABEL org.opencontainers.image.revision=$KADAN_COMMIT
EXPOSE 8080
