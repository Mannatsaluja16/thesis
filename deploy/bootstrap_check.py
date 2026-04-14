"""Run once to verify torch, set up nginx + systemd, and start the Flask app on EC2."""
import io
import paramiko

NGINX_CONF = """
server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
"""

SYSTEMD_UNIT = """
[Unit]
Description=Fault Tolerance Flask App
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/local/bin/gunicorn -w 2 -b 127.0.0.1:5000 src.cloud_controller.api_gateway:app
Restart=always
RestartSec=5
Environment=PYTHONPATH=/home/ec2-user/app

[Install]
WantedBy=multi-user.target
"""

key = paramiko.RSAKey.from_private_key_file("F:/mannat_thesis/cloud-key-pair.pem")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("34.235.165.89", username="ec2-user", pkey=key, timeout=30)

# Write config files via SFTP to /tmp, then sudo-move them
sftp = c.open_sftp()
sftp.putfo(io.BytesIO(NGINX_CONF.encode()), "/tmp/fault_tolerance.conf")
sftp.putfo(io.BytesIO(SYSTEMD_UNIT.encode()), "/tmp/fault_tolerance_app.service")
sftp.close()

def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=30)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    if out: print(out)
    if err: print("ERR:", err)

run("python3 -c 'import torch; print(\"torch\", torch.__version__)'")
run("df -h / | tail -1")
run("sudo cp /tmp/fault_tolerance.conf /etc/nginx/conf.d/fault_tolerance.conf")
run("sudo nginx -t 2>&1")
run("sudo systemctl enable nginx --quiet && sudo systemctl restart nginx && echo 'nginx OK'")
run("sudo cp /tmp/fault_tolerance_app.service /etc/systemd/system/fault_tolerance_app.service")
run("sudo systemctl daemon-reload")
run("sudo systemctl enable fault_tolerance_app --quiet && sudo systemctl restart fault_tolerance_app")
run("sleep 3 && systemctl is-active fault_tolerance_app")
run("curl -s http://localhost/metrics/summary")

c.close()
