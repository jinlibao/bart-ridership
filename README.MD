# COSC 5010 Final Project — Data Mining in Public Transportation

[![DOI](https://zenodo.org/badge/131897531.svg)](https://zenodo.org/badge/latestdoi/131897531)

Please download the data from [BART-Ridership](https://www.kaggle.com/saulfuh/bart-ridership/data), and put the .csv files under the folder `./data/bart-ridership`.

The folder structure should be like the following

    .
    ├── data
    │   └── bart-ridership
    │       ├── date-hour-soo-dest-2016.csv
    │       ├── date-hour-soo-dest-2017.csv
    │       └── station_info.csv
    └── program
        ├── bart.ipynb
        ├── bart.py
        └── output
            ├── bart_overview_1.pdf
            ├── bart_overview_2.pdf
            ├── bart_routes.pdf
            └── bart_throughput.pdf

To execute the Python program, just go the project folder and then type the following script in the shell:

```Shell
cd program
./bart.py
```

