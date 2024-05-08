import http.server
import socketserver
import os
import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa

PORT_HTTP = 8080
CERTS_TO_ISSUE = 2  # Maximum number of certificates to issue
issued_count = 0  # Counter for issued certificates

class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global issued_count
        query = self.path[1:]  # Extract endpoint from URL path

        if query == "sign" and issued_count < CERTS_TO_ISSUE:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Load CA cert and key
            with open('ca.crt', 'rb') as f:
                ca_cert_pem = f.read()
            with open('ca.key', 'rb') as f:
                ca_key_pem = f.read()

            # Sign the CSR
            signed_cert_pem = self.sign_csr(post_data, ca_cert_pem, ca_key_pem)

            self.send_response(200)
            self.send_header('Content-type', 'application/x-pem-file')
            self.end_headers()
            self.wfile.write(signed_cert_pem)

            issued_count += 1
            if issued_count >= CERTS_TO_ISSUE:
                print("Issued the designated number of certificates, terminating server.")
                os._exit(0)
        elif query == "getca":
            self.send_response(200)
            self.send_header('Content-type', 'application/x-pem-file')
            self.end_headers()
            with open('ca.crt', 'rb') as f:
                self.wfile.write(f.read())

    def sign_csr(self, csr_pem, ca_cert_pem, ca_key_pem):
        csr = x509.load_pem_x509_csr(csr_pem, default_backend())
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem, default_backend())
        ca_key = serialization.load_pem_private_key(ca_key_pem, password=None, backend=default_backend())

        builder = x509.CertificateBuilder()
        builder = builder.subject_name(csr.subject)
        builder = builder.issuer_name(ca_cert.subject)
        builder = builder.public_key(csr.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.today() - datetime.timedelta(days=1))
        builder = builder.not_valid_after(datetime.datetime.today() + datetime.timedelta(days=365))

        # Copy all extensions from CSR to the new certificate
        for extension in csr.extensions:
            builder = builder.add_extension(extension.value, extension.critical)

        certificate = builder.sign(
            private_key=ca_key, algorithm=hashes.SHA256(), backend=default_backend()
        )

        return certificate.public_bytes(serialization.Encoding.PEM)

# HTTP server setup
http_handler = CustomHTTPHandler
httpd = socketserver.TCPServer(("", PORT_HTTP), http_handler)
print("Serving HTTP at port", PORT_HTTP)

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    httpd.shutdown()
