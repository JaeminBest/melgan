import boto3
import os
from requests_aws4auth import AWS4Auth
#import cottoncandy as cc

# AWS S3 setting
service = 's3'
session = boto3.Session(
  aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''),
  aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')
)
credentials = session.get_credentials()

region = os.environ.get('AWS_REGION', '')
http_auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service)

s3 = boto3.resource('s3')


# cci = cc.get_interface(os.environ.get('S3_BUCKET',''),
#                         ACCESS_KEY=os.environ.get('AWS_ACCESS_KEY_ID',''),
#                         SECRET_KEY=os.environ.get('AWS_SECRET_ACCESS_KEY',''),
#                         endpoint_url=os.environ.get('S3_ENDPOINT',''))