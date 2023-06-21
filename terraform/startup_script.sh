#!/bin/bash
echo "Made it to the user_data"
mv /home/ubuntu/linc-cv /home/ubuntu/linc-cv-backup
aws s3 cp s3://linc-backup/linc-cv/linc-cv-20181223.tar.gz /home/ubuntu/linc-cv-backup/linc-cv-20181223.tar.gz
sudo supervisorctl stop all
sudo supervisorctl status
scp -r /path/to/updated/linc-cv ubuntu@current_instance_ip:/home/ubuntu/
sudo supervisorctl start all
sudo supervisorctl status