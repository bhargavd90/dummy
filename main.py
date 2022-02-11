from flask import Flask, request
from flask_cors import CORS
import cluster
import warnings

warnings.simplefilter("ignore")

app = Flask(__name__)
CORS(app)


# cluster.storeHierarchyData()
# cluster.generateHierarchy(["content"], 2, 0, 1)


@app.route('/generate_hierarchy')
def generate_hierarchy():
    try:
        split_entity_string = request.args.get('split_entity_string')
        content_depth_needed = int(request.args.get('content_depth_needed'))
        content_capture_needed = float(request.args.get('content_capture_needed'))
        time_place_weight = float(request.args.get('time_place_weight'))
        content_weight = float(request.args.get('content_weight'))
        topic_interest_keyword = request.args.get('topic_interest_keyword')
        from_date_keyword = request.args.get('from_date_keyword')
        to_date_keyword = request.args.get('to_date_keyword')
        split_entity_list = [s.lower() for s in split_entity_string.split(":")]
        cluster.generateHierarchy(split_entity_list, content_depth_needed, content_capture_needed, time_place_weight, content_weight, topic_interest_keyword, from_date_keyword, to_date_keyword)
        return 'success'
    except Exception as err:
        print("error while generating hierarchy : ", err)
        return str(err)


@app.route('/search_node')
def search_node():
    try:
        search_term = request.args.get('search_term')
        nodeId = cluster.search_node(search_term)
        return nodeId
    except Exception as err:
        print("error while searching for a news cluster : ", err)
        return str(err)


app.run(host='127.0.0.1', port='5000')
