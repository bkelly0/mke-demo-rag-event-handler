$event = @{
  type = "google.cloud.storage.object.v1.finalized"
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
    -Headers @{
    "ce-type" = "google.cloud.storage.object.v1.finalized"
  } `
  -Body $body