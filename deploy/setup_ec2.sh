#!/bin/bash
set -e

echo "=== Setting up EC2 instance for Fault Tolerance Framework ==="

# Update system
sudo yum update -y
sudo yum install -y python3 python3-pip nginx git

# Install Gunicorn
pip3 install gunicorn

# App directory
APP_DIR="/home/ec2-user/app"
sudo mkdir -p $APP_DIR
sudo chown ec2-user:ec2-user $APP_DIR

# Install Python dependencies
pip3 install -r $APP_DIR/requirements.txt

# Create necessary directories
mkdir -p $APP_DIR/data/raw $APP_DIR/data/processed $APP_DIR/data/synthetic
mkdir -p $APP_DIR/models $APP_DIR/results/plots $APP_DIR/results/reports

# Configure Nginx
sudo cp $APP_DIR/deploy/nginx.conf /etc/nginx/conf.d/fault_tolerance.conf
sudo nginx -t && sudo systemctl enable nginx && sudo systemctl start nginx

# Configure systemd service
sudo cp $APP_DIR/deploy/fault_tolerance_app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable fault_tolerance_app
sudo systemctl start fault_tolerance_app

echo "=== EC2 setup complete ==="
sudo systemctl status fault_tolerance_app --no-pager
