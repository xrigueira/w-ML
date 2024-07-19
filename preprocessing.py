import os
import logging

from preprocessors.checkGaps import checkGaps
from preprocessors.filler import filler
from preprocessors.joiner import joiner
from preprocessors.labeler import labeler
from preprocessors.filterer import mfilterer
from preprocessors.smoother import smoother

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""Preprocessing starts from the original univariate txt files."""

# Define the data we want to study
files = [f for f in os.listdir("raw_data") if os.path.isfile(os.path.join("raw_data", f))]

varNames = [i[0:-4] for i in files] # Extract the names of the variables
stations = [901, 905, 907] # Define with stations to process

# Define the time frame we want to use (a: months (not recommended), b: weeks, c: days). 
timeFrame = 'c'
timeStep = '15 min' # '1 day', '15 min'

if __name__ == '__main__':

    for varName in varNames:
    
        # Find the gaps in the time series, add them and leave them blank
        checkGaps(File=f'{varName}.txt', timestep=timeStep, varname=varName)
        logging.info(f'checkGaps() {varName} DONE')

        # Add 31s if timestep is 'a' and week information. See filler.py for details
        filler(File=f'{varName}_full.csv', timeframe=timeFrame, timestep=timeStep, varname=varName)
        logging.info(f'filler() {varName} DONE')
    
    for station in stations:
        
        # Join the filled databases
        joiner(station=station)
        logging.info(f'joiner() {station} DONE')

    # Add the Label data to all stations
    labeler(stations=stations)
    logging.info('labeler() DONE')
    
    for station in stations:

        # Filter out those months or weeks or days (depending on the desired
        # time unit) with too many NaN in several variables and iterate on the rest
        mfilterer(File=f'labeled_{station}.csv', timeframe=timeFrame, timestep=timeStep)
        logging.info(f'mfilterer() {station} DONE')

        # Smooth the data
        smoother(station=station)
        logging.info(f'smoother() {station} DONE')
