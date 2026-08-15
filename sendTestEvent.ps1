$event = @{
  specversion = "1.0"
  id = "local-test-001"
  source = "//storage.googleapis.com/projects/_/buckets/bkelly-mke-rag-data"
  type = "google.cloud.storage.object.v1.finalized"
  subject = "objects/CH295-sub1.pdf"
  time = "2026-08-14T12:00:00Z"
  datacontenttype = "application/json"
  data = @{
    bucket = "bkelly-mke-rag-data"
    name = "CH295-sub1.pdf"
    generation = "1"
    metageneration = "1"
    size = "1"
  }
}

$body = $event | ConvertTo-Json -Depth 3

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body