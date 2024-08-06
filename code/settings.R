################################################################################
# Settings and initialization script                                           #
# Joel Gastelaars                                                              #
# September 2021                                                               #
################################################################################

#### 1. Load functions ####
source('code/functions.R')

#### 2. Load packages and initialize other settings ####
# Options (disable scientific notation)
options(scipen = 999)

lib <- c("tidyr", "dplyr", "jsonlite", "stringr", "data.table", "tdplyr", "dbplyr", "httr", "lubridate", "stringi",
         "ggplot2", "plotly", "cronR", "shinyFiles", "miniUI", "shiny")
packages(lib)


