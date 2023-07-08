# Best practices

## Commands

Assuming docker compose including localstack started, create new S3 bucket

```
aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration
make_bucket: nyc-duration
```
