variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for linc-cv"
  default     = "ami-07c6a705483a25f67"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "r5.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  default     = "linc-cv"
}
