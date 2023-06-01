#%% 
import labelbox
import json

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGljd2VqY2QwYWRoMDcwMzFpMDgwN2QyIiwib3JnYW5pemF0aW9uSWQiOiJjbGljd2VqYzAwYWRnMDcwM2dyMzRkM2pjIiwiYXBpS2V5SWQiOiJjbGljeHBlNHkwNGY3MDd5cmFzd2phZXQ1Iiwic2VjcmV0IjoiODY3YjM1NTllMDA4ZmQ3NDEyODgwODY3NzkwNTdjOWQiLCJpYXQiOjE2ODU2MTE3MzQsImV4cCI6MjMxNjc2MzczNH0.e04oD4kfP3-2z3cranyFcYRqAoe98XyjhkupmXauo-U"
project_id = "clicwkp7q06t607zg87vgerom"

#%% exporting the json file 

client = labelbox.client.Client(api_key=api_key)

project = client.get_project(project_id)

labels = project.export_v2(params={
	"data_row_details": True,
	"metadata": True,
	"attachments": True,
	"project_details": True,
	"performance_details": True,
	"label_details": True,
	"interpolated_frames": True
  })

labels.wait_till_done()

if labels.errors:
  print(labels.errors)

json_list = labels.result
# print("results: ", json_list)

#%% Write the json file

i=0
for element in json_list:
  json_object = json.dumps(element)
  with open("input/sample"+str(i)+".json", "w") as outfile:
    outfile.write(json_object)
  i+=1

# from main import main
# main()




