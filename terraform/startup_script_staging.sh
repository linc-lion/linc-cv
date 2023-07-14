#!/bin/bash
sudo supervisorctl stop all
sudo supervisorctl status
mv /home/ubuntu/linc-cv/ /home/ubuntu/linc-cv-backup
cd /home/ubuntu/
git clone https://github.com/linc-lion/linc-cv
git checkout staging
wget -P /home/ubuntu/linc-cv/linc_cv/ https://github.com/linc-lion/linc-cv-data/releases/latest/download/linc-cv-data.tar.gz
cd /home/ubuntu/linc-cv/linc_cv/
tar -xvzf /home/ubuntu/linc-cv/linc_cv/linc-cv-data.tar.gz
sudo supervisorctl start all
sudo supervisorctl status