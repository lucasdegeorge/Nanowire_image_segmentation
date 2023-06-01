#%% 
import labelbox
import numpy as np
from PIL import Image

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGljd2VqY2QwYWRoMDcwMzFpMDgwN2QyIiwib3JnYW5pemF0aW9uSWQiOiJjbGljd2VqYzAwYWRnMDcwM2dyMzRkM2pjIiwiYXBpS2V5SWQiOiJjbGljeHBlNHkwNGY3MDd5cmFzd2phZXQ1Iiwic2VjcmV0IjoiODY3YjM1NTllMDA4ZmQ3NDEyODgwODY3NzkwNTdjOWQiLCJpYXQiOjE2ODU2MTE3MzQsImV4cCI6MjMxNjc2MzczNH0.e04oD4kfP3-2z3cranyFcYRqAoe98XyjhkupmXauo-U"
project_id = "clicwkp7q06t607zg87vgerom"
# dataset_id = 

#%% 

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

export_json = labels.result
print("results: ", export_json)



#%% 

for label in labels.labels:
    # Assuming the label contains a segmentation mask
    mask = label.data.get('mask')

    if mask:
        # Convert the mask to a binary numpy array
        mask_array = np.array(mask, dtype=np.uint8)

        # Create a PIL Image from the binary mask array
        mask_image = Image.fromarray(mask_array * 255)

        # Save the mask image as a PNG file
        mask_image.save(f'{label.uid}.png')
