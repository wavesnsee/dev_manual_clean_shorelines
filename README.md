# dev_manual_clean_shorelines

With this project, we can do 2 different things: 
- Manually create shorelines with a GUI ('dev_manual_creation_shorelines.py')
- Clean them with a GUI (this is deprecated now, as editing of shorelines is done automatically now) ('dev_manual_clean_shoreline.py').

Here are some details to run manual cleaning of shorelines:

Arguments of the program:
- Input_dir (coastlines)
- Output_dir (selected coastlines)
- Images_dir (Directory containing average images 'A_CAM*fps_*s_*.jpg')

During execution, the user will have the following choices to process the shoreline:
- Keep the shoreline as it is (if lucky !) 
- Throw away the shoreline (dirty case) 
- Remove any part(s) of the shoreline
- Choose any part(s) of the shoreline


