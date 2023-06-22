variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for linc-cv"
  default     = "ami-0a9ca2a242d7fc4df"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "t3.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  default     = "linc-cv"
}
