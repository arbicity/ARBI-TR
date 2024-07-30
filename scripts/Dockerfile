FROM alpine:latest
RUN apk --no-cache add openssl
WORKDIR /app
COPY generate-certificates.sh /app/
RUN chmod +x /app/generate-certificates.sh