import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCOUNT_ID        = os.getenv("AWS_ACCOUNT_ID")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_AMI_ID            = os.getenv("AWS_AMI_ID", "ami-098e39bafa7e7303d")
AWS_INSTANCE_TYPE     = os.getenv("AWS_INSTANCE_TYPE", "t3.micro")
AWS_KEY_PAIR_NAME     = os.getenv("AWS_KEY_PAIR_NAME", "cloud-key-pair")
GITHUB_REPO           = os.getenv("GITHUB_REPO")

RANDOM_SEED = 42
WINDOW_SIZE = 10
INPUT_SIZE  = 5
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
FAULT_THRESHOLD = 0.5
REPLICATION_THRESHOLD = 0.75
