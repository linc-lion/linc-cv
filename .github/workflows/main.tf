provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "ec2_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name

  tags = {
    Name = "MyEC2Instance"
  }
}

  user_data = <<-EOF
    #!/bin/bash
    ssh -i ~/my_pem.pem ubuntu@current_instance_ip
    mv /home/ubuntu/linc-cv /home/ubuntu/linc-cv-backup
    aws s3 cp s3://linc-backup/linc-cv/linc-cv-20181223.tar.gz /home/ubuntu/linc-cv-backup/linc-cv-20181223.tar.gz
    sudo supervisorctl stop all
    sudo supervisorctl status
    scp -r /path/to/updated/linc-cv ubuntu@current_instance_ip:/home/ubuntu/
    sudo supervisorctl start all
    sudo supervisorctl status
  EOF