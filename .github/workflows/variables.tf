variable "aws_region" {
  description = "AWS region"
  default     = "us-west-2"
}

variable "ami_id" {
  description = "AMI ID"
  default     = "ami-xxxxxxxx"
}

variable "instance_type" {
  description = "EC2 instance type"
  default     = "t2.xlarge"
}

variable "key_name" {
  description = "EC2 key pair name"
  default     = "my-key-pair"
}
