# target_audience_newsletter
The repo finds cyclists who usually go on rides close to each other, and assigns them a starting point for an upcoming group ride. 

It contains 4 files:
- **assign_starting_points.ipynb** - a Jupyter notebook that analysis the cyclists, how often they ride, whether they start from the same location or not, etc. The notebook also contains the code that breaks down the area from the dataset into hexagons and assigns a cyclist to a hexagon. The centers of the hexagons users belong to are the starting points for the group ride.
- **assign_starting_points.py** - a script that executes the csv file ingestion, executes the code that assigns cyclists to a starting point, and produces a csv file that contains a list of potential cyclists
a specific cyclist can ride with
- **tour.csv** - a synthetic sample dataset that contains starting points for the cyclists' rides (latitude and longitude)
- **result.csv** - a csv file that lists potential cyclists for a given user, the starting point, and the location of that cyclist's starting point

To test this code on your dataset, execute the following line from a location that contains this code locally: ./assign_starting_points.py --input_file {your_dataset_name.csv}. For example: ./assign_starting_points.py --input_file tours.csv produces the results.csv file
 

