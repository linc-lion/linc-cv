variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for linc-cv"
  default     = "ami-02f53e5bce1535b6c"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "t3.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  default     = "thomas-dev"
}
