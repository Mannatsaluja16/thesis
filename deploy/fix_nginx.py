"""Remove the default server block from nginx.conf so our conf.d proxy config takes over."""
import io
import paramiko

NGINX_CONF = (
    "user nginx;\n"
    "worker_processes auto;\n"
    "error_log /var/log/nginx/error.log notice;\n"
    "pid /run/nginx.pid;\n"
    "include /usr/share/nginx/modules/*.conf;\n"
    "\n"
    "events {\n"
    "    worker_connections 1024;\n"
    "}\n"
    "\n"
    "http {\n"
    "    access_log  /var/log/nginx/access.log;\n"
    "    sendfile on;\n"
    "    tcp_nopush on;\n"
    "    keepalive_timeout 65;\n"
    "    types_hash_max_size 4096;\n"
    "    include /etc/nginx/mime.types;\n"
    "    default_type application/octet-stream;\n"
    "    include /etc/nginx/conf.d/*.conf;\n"
    "}\n"
)

key = paramiko.RSAKey.from_private_key_file("F:/mannat_thesis/cloud-key-pair.pem")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("34.235.165.89", username="ec2-user", pkey=key, timeout=30)

sftp = c.open_sftp()
sftp.putfo(io.BytesIO(NGINX_CONF.encode()), "/tmp/nginx.conf")
sftp.close()


def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=15)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out:
        print(out)
    if err:
        print("ERR:", err)


run("sudo cp /tmp/nginx.conf /etc/nginx/nginx.conf")
run("sudo nginx -t 2>&1")
run("sudo systemctl restart nginx && echo 'nginx restarted'")
run("curl -s http://localhost/metrics/summary")
c.close()
