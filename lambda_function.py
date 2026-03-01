import json, os, time, boto3

runtime = boto3.client("sagemaker-runtime")
dynamo = boto3.resource("dynamodb").Table(os.environ["LOG_TABLE"])
ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]

def safe_json(value):
    try:
        return json.dumps(value)
    except:
        return str(value)

def _parse_body(event):
    """
    Support:
      - API Gateway proxy:  {"body": "{ \"inputs\": \"...\" }"}
      - Direct invoke:      {"inputs": "..."}
    """
    if "body" in event:
        raw = event["body"]
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        elif isinstance(raw, dict):
            return raw
        else:
            return {}
    else:
        # direct invoke / test event with inputs at top level
        return event if isinstance(event, dict) else {}


def lambda_handler(event, context):
    body = _parse_body(event)
    text = body.get("inputs", "")

    # Hard guard so you immediately see if something is wrong
    if not text.strip():
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Empty 'inputs' received by Lambda",
                "raw_event_sample": str(event)[:500]
            })
        }

    payload = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": 128,
            "max_length": 256,
            "temperature": 0.0,
            "top_p": 0.9
        }
    }

    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(resp["Body"].read().decode())

    log_item = {
            "request_id": f"{int(time.time()*1000)}#{context.aws_request_id}",
            "prompt": text,
            "response": safe_json(result),
            "timestamp": str(int(time.time()))
        }
    dynamo.put_item(Item=log_item)

    return {
        "statusCode": 200,
        "body": json.dumps({"result": result})
    }