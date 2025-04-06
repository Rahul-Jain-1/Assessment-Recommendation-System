__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from rag_pipeline import generate_shl_assessment

def respond(data,  err=None):
    if err is not None:
        return {
            "statusCode": "500",
            "error": str(err),
            "data":"",
            "status":"failed"
        }
    return {"statusCode": "200", 
            "error":"",
            "data": data, 
            "status": "success"}


def lambda_handler(event, context):
    try:
        event_body = event["params"]["querystring"]
        query = event_body["query"]
       
        result = generate_shl_assessment(query)
        return respond(result)

    except Exception as e:
        return respond("", e)


# event = {
#     "params":{
#         "querystring": {
#             "query":""   
#         }
#     }
# }

# print(lambda_handler(event, ""))

