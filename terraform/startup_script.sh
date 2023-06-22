#!/bin/bash
echo "Made it to the user_data 12345"
echo "$PWD"
sudo supervisorctl stop all
sudo supervisorctl status
mv /home/ubuntu/linc-cv/ /home/ubuntulinc-cv-backup
cd /home/ubuntu/
git clone https://github.com/linc-lion/linc-cv
sudo apt-get upgrade -y
sudo apt-get -f install -y
sudo apt-get install ec2-instance-connect -y
sudo apt install awscli -y
aws s3 cp s3://linc-model-artifact/linc-cv/20230619/ /linc-cv/linc-cv/
cd /linc-cv/linc-cv/
tar -xvzf data.tar.gz
echo "$PWD"
sudo supervisorctl start all
sudo supervisorctl status