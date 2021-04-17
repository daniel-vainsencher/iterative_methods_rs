error_message = """ 

        You do not have the Python modules needed to generate the visualizations for this example.
        You can install the dependencies or run the example without visualizations.

        To install the dependencies use the following steps:

            0) If you don't already have it, install Python3 following the instructions at https://www.python.org/downloads/.
            
            1) Install pip and virtual env according to the instructions here:
            
            https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#:~:text=Installing%20virtualenv&text=venv%20is%20included%20in%20the,Python%20packages%20for%20different%20projects.

            2) Set up a virtual environment that will contain the dependencies:
            `$ virtualenv <name>`
            
            3) Activate the environment:
            `$ source <name>/bin/activate`

            4) Install the requirements using the requirements.txt file:
            `$ pip install -r ./visualizations_python/requirements.txt`

            5) Rerun the examples.

        To run the example without visualizations, add the command line argument 'false':

            `$ ./target/debug/examples/reservoir_histogram_animation false`
        """