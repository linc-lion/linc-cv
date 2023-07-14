terraform {
  backend "s3" {
    bucket = "linc-terraform"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}