################################################################################
# Template main script which sources the relevant scripts                      #
# Joël Gastelaars                                                              #
# Sept 2021                                                                    #
################################################################################

# TODO: this is an example script. Only keep what you need.
# Basic setup
# TODO: 1. It is important to source the functions and init settings from another script to keep your main script clean
# TODO: 2. If needed, create a configuration file, see example in config.yml
# TODO: 3. Always print start/end times for your reference and to help estimate computational time.
# Adding them between steps can help you optimize the big chunks first.

#### Utilization (settings, packages, functions) ####
cat(as.character(Sys.time()), "Started script. \n")
# Enforce the proper working directory for scheduled CRONjob and local jobs
setwd("~/github/template_repo")
source("code/functions.R")
source("code/settings.R")

#### Set config ####
# CONFIG SETTINGS: default, test, eda, dashboard, load_all, nps, nps_load
Sys.setenv(R_CONFIG_ACTIVE = 'nps')
config_settings <<- config::get(file = Sys.getenv("R_CONFIG_FILE", "config.yml"))

#### Run script --------------------------------------------------------------
# Extract survey, questionnaire and response data with API calls to GrowPromoter
if (config_settings$parse) {
  cat(as.character(Sys.time()), "Extracting data \n")
  source('code/extractData.R')
  cat(as.character(Sys.time()), "Preparing data \n")
  source('code/prepData.R')
}

# Write data back to DWH
if (config_settings$writeback) {
  cat(as.character(Sys.time()), "Writeback to DWH. \n")
  source('code/outputDWH.R')
}
cat(as.character(Sys.time()), "Finished script. \n")

# Clean up environment except a,b,c
rm(list = setdiff(ls(), c("a", "b", "c")))
gc()
