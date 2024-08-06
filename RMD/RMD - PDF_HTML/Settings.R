################################################################################
# Settings & Functions                                                         #
# Joel Gastelaars                                                              #
# November 2021                                                                #
################################################################################

#### 1. Load packages and init ####
# Set a CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Automatically install and load packages from LMN-CRAN
packages <- function(pkg) {
  new_pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new_pkg)) {
    install.packages(new_pkg, dependencies = TRUE, repos = "https://rsp.lmn.nl:4343/LMN-Cran/__linux__/focal/latest")
  }
  sapply(pkg, require, character.only = TRUE)
}

# list of used packages
lib <- c("tidyr", "jsonlite","zoo", "stringr", "readr", "data.table", "DT", "httr", "lubridate", "stringi",
         "ggplot2", "miniUI", "knitr", "magrittr", "tidyverse", "corrplot",
         "scales", "psych", "labelled", "readxl","readr", "zoo", 
         "correlationfunnel","prettydoc","alluvial","ggalluvial","kableExtra", "webshot2","htmltools", "bookdown",
         "fancyhdr", "graphicx", "odbc", "DBI", "dbplyr", "dplyr", "tseries", "forecast",
         "CausalImpact", "webr", "glue"
         # "naniar", "plotly", "cronR", "shinyFiles", "pracma","DataExplorer", "dataMaid",  "config", "skimr", "vcd", "cluster",
         # "factoextra", "NbClust", "parallel", "infer", "timetk", "shiny", "shinydashboard", "networkD3"
         )
# load packages
packages(lib)

# Options (disable scientific notation)
options(scipen = 999)

#### 2. Functions ####
# Easy to use negation of %in%
`%notin%` <- Negate(`%in%`)

# Turns datetime into seconds since epoch (UNIX timestamp).
# Used to filter survey responses in the GP API call
# Insert x as date in datetime format (for example as_datetime("2021-08-20 17:00:00")).
time_since_epoch <- function(x) {
  x2 <- lubridate::ymd_hms(format(x, tz = "GMT", usetz = FALSE))
  epoch <- lubridate::ymd_hms("1970-01-01 00:00:00")
  time_since_epoch <- (x2 - epoch) / dseconds()
  return(time_since_epoch)
}

# prep data for rolling year, can also be used after filtering out members that have filled multiple surveys
prep_data_for_plot <- function(df) {
                                    setDT(df)
                                    #using zoo package for rolling sum
                                    df_zoo_tel      <- zoo(df$tel)
                                    df_zoo_score    <- zoo(df$score)
                                    tel_cum         <- as.data.frame(rollapply(df_zoo_tel, 12, sum, align = "right", fill = NA) )
                                    score_cum       <- as.data.frame(rollapply(df_zoo_score, 12, sum, align = "right", fill = NA) )
                                    ry_df           <- cbind(df, tel_cum, score_cum)
                                    names(ry_df)[6] <- "tel_cum"
                                    names(ry_df)[7] <- "score_cum"
                                    ry_df           <- ry_df %>% mutate(ry_avg_score = round(score_cum/tel_cum,2))
                                    ry_df
                                    }



# function to change color of rmd depending on output
colorize <- function(x, color) {
  if (knitr::is_latex_output()) {
    sprintf("\\textcolor{%s}{%s}", color, x)
  } else if (knitr::is_html_output()) {
    sprintf("<span style='color: %s;'>%s</span>", color,
            x)
  } else x
}

# connection to db

ConToAzure <- function(env = "dev", db = "ds"){
  # Function to connect to Azure.
  # ! Requirement: You must define certain variables in your .Renviron file:
  #   az_driver
  # az_port
  # az_server_dev
  # az_db_dev
  # az_user_dev
  # az_pw_dev
  # * also include equivalent variables for tst, acc and prd
  # - env: environment you want to connect to: "dev" (default), "tst", "acc" or "prd"
  # - db: database you want to connect to: "ds" (default: datascience), "ldm" (ldm)
  
  max_tries <- 10
  current_try <- 1
  success <- FALSE
  
  env_Driver = Sys.getenv("az_driver")
  env_port = Sys.getenv("az_port")
  env_Server = Sys.getenv(paste0("az_server_", env, "_", db))
  env_Database = Sys.getenv(paste0("az_db_",env, "_", db))
  env_UID = Sys.getenv("az_user")
  env_PWD = Sys.getenv(paste0("az_pw_", env, "_", db))
  
  while(current_try <= max_tries && !success){
    con <- try({dbConnect(odbc(),
                          Driver = env_Driver,
                          Server = env_Server,
                          Database = env_Database,
                          UID = env_UID,
                          PWD = env_PWD,
                          port = env_port,
                          timeout = 90)},
               silent=TRUE)
    
    
    if(class(con) != "try-error"){
      success <- TRUE
      print("Succesfully connected")}
    else{
      print("Failed to connect. Retrying...")
      Sys.sleep(5)
    }
    current_try <- current_try + 1
  }
  
  return(con)
}


# Function to format numbers and dates for inline text in RMarkdown
inline_numbers <- function(x) {
  if (is.numeric(x)) {
    if (abs(x - round(x)) < .Machine$double.eps) {
      # Treat as integer: format with zero digits after decimal, use thousands separator
      formatted <- format(x, digits = 0, big.mark = ",")
    } else {
      # Treat as floating-point number: format with two digits after decimal, use thousands separator
      formatted <- format(x, digits = 2, nsmall = 2, big.mark = ",")
    }
  } else if (inherits(x, "Date")) {
    # Treat as date: format as "Month day, year"
    formatted <- format(x, "%B %d, %Y")
  } else {
    # If not numeric or date, return as is
    formatted <- x
  }
  
  # Wrap formatted result in bold markdown
  paste0("**", formatted, "**")
}

knit_hooks$set(inline = inline_numbers)

#format numbers in plots
format_labels_perc <- function(x) {
  sprintf("%.1f", round(x, 1))
}

format_labels_int <- function(x) {
  format(round(x), big.mark = ",", scientific = FALSE)
}

# alternative Lydia
# Load necessary packages
library(dplyr)
library(scales)

# Define the function to format numbers
format_numbers <- function(df) {
  
  # Helper function to format percentages
  format_labels_perc <- function(x) {
    sprintf("%.1f", round(x, 1))
  }
  
  # Helper function to format integers
  format_labels_int <- function(x) {
    format(round(x), big.mark = ",", scientific = FALSE)
  }
  
  # Helper function to format large numbers
  format_large_numbers <- function(x) {
    if (is.na(x)) {
      return(NA)
    } else if (x >= 1e6) {
      return(paste0(round(x / 1e6, 1), "M"))
    } else if (x >= 1e5) {
      return(paste0(round(x / 1e3, 1), "K"))
    } else {
      return(format_labels_int(x))
    }
  }
  
  # Initialize list to store formatted columns
  formatted_columns <- list()
  
  # Iterate through numeric columns to format them
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      if (all(df[[col]] < 1 & df[[col]] > 0, na.rm = TRUE)) {
        formatted_columns[[paste0(col, "_rounded")]] <- format_labels_perc(df[[col]])
      } else {
        formatted_values <- sapply(df[[col]], format_large_numbers)
        formatted_columns[[paste0(col, "_rounded")]] <- formatted_values
        
        # Assign the correct suffix
        if (any(grepl("M", formatted_values))) {
          col_suffix <- "_rounded_M"
        } else if (any(grepl("K", formatted_values))) {
          col_suffix <- "_rounded_K"
        } else {
          col_suffix <- "_rounded"
        }
        
        names(formatted_columns)[names(formatted_columns) == paste0(col, "_rounded")] <- paste0(col, col_suffix)
      }
    }
  }
  
  # Convert the formatted columns list to a data frame
  if (length(formatted_columns) > 0) {
    formatted_df <- as.data.frame(formatted_columns)
    # Combine the original dataframe with the formatted columns
    df <- bind_cols(df, formatted_df)
  }
  
  return(df)
}

# # Example usage with products_df
# formatted_df <- format_numbers(products_df)
# print(formatted_df)
