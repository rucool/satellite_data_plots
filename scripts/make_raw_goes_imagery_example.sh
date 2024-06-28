#!/bin/bash

# Source the global bashrc
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

# Source the local bashrc
if [ -f ~/.bashrc ]; then
. ~/.bashrc
fi

TOOLBOX_DIR=/path/to/satellite_data_plots/
IMG_DIR=/path/to/base/image/directory/
BATHY_DIR=/path/to/directory/with/bathymetry/files/

conda activate satellite_data_plots

python ${TOOLBOX_DIR}/scripts/goes_web_plots.py -f ${TOOLBOX_DIR}/files/web_regions.yml -d ${IMG_DIR} -b $BATHY_DIR -s ${TOOLBOX_DIR}/files/standardized_variable_names.yml