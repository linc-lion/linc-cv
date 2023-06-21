provider "aws" {
  region  = var.aws_region
}

resource "aws_instance" "ec2_instance" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name

  tags = {
    Name = "Linc-CV-Prod-06-2023"
  }
    user_data = file("startup_script.sh")
}

