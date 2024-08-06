################################################################################
# R Functions for LMN                                                          #
# Joel Gastelaars                                                              #
# November 2021  | last update: april 2022                                     #
################################################################################

# Automatically install and load packages from LMN-CRAN
packages <- function(pkg) {
  new_pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new_pkg)) {
    install.packages(new_pkg, dependencies = TRUE, repos = "https://rsp.lmn.nl:4343/LMN-Cran/__linux__/focal/latest")
  }
  sapply(pkg, require, character.only = TRUE)
}

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

# TODO: change to LMN color scheme
# Place your color coding scheme here for the plots below
lmn_colors <- c("#0F238C", "#FF6600", "#BFC2C5", "#7f7f7f", "#000000")

# # Define colors for my plots with define_palette
# lmn <- define_palette(
#   swatch = lmn_colors,
#   gradient = c(lower = lmn_colors[1], upper = lmn_colors[2]), # add upper and lower colors for continuous colors
#   line = c("#7f7f7f", "#7f7f7f"),
#   gridline = "#000000")
# # Set theme
# ggthemr(lmn)

# Function to plot a nice barplot with plotly, for e.g. age, participation, saldo and partners
plotly_barplot <- function(dataframe, variable) {
  ggplotly(dataframe %>% 
             group_by(.data[[variable]]) %>%
             summarise(n = n(), perc = n/nrow(dataframe)) %>%
             ggplot(aes(x = .data[[variable]], y = n, fill = perc)) +
             theme_bw() +
             theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1)) +
             geom_col())
}

# Plot univariate analysis using plotly
# Input is dataframe containing variable to plot and target variable
uni_plot <- function(dataframe, variable, target) {
  avg_target <- dataframe %>%
    filter(!is.na(.data[[variable]])) %>%
    summarise(average = mean(as.numeric(as.character(.data[[target]]))))
  
  dataframe %>%
    filter(!is.na(.data[[variable]])) %>%
    group_by(group = .data[[variable]]) %>%
    summarise(Percentage = mean(as.numeric(as.character(.data[[target]]))),
              Total = n()) %>%
    cbind(avg_target) %>% 
    plot_ly() %>%
    add_trace(x = ~group, y = ~Total, type = "bar", alpha = 0.6,
              color = I(lmn_colors[1]), name = "Total") %>%
    add_trace(x = ~group, y = ~Percentage, type = "scatter", mode = "linesx", alpha = 0.8,
              color = I(lmn_colors[2]), name = "Percentage xx", yaxis = "y2") %>%
    add_trace(x = ~group, y = ~average, type = "scatter", mode = "linesx", alpha = 0.4,
              color = I(lmn_colors[5]), name = "Avg percentage xx", yaxis = "y2") %>%
    layout(yaxis = list(side = "left", title = "Total"),
           yaxis2 = list(side = "right", overlaying = "y", title = "Percentage xx"),
           xaxis = list(title = variable))
}

# Plot univariate analysis on logarithmic scale using plotly. 
# Input is dataframe containing variable to plot and target variable
uni_lnplot <- function(dataframe, variable, target) {
  avg_target <- dataframe %>%
    filter(!is.na(.data[[variable]])) %>%
    summarise(average = mean(as.numeric(as.character(.data[[target]]))))
  
  dataframe %>%
    filter(!is.na(.data[[variable]])) %>%
    group_by(group = round(log(.data[[variable]] + 1))) %>%
    summarise(Percentage = mean(as.numeric(as.character(.data[[target]]))),
              Total = n()) %>%
    cbind(avg_target) %>% 
    plot_ly() %>%
    add_trace(x = ~group, y = ~Total, type = "bar", alpha = 0.6,
              color = I(lmn_colors[1]), name = "Total") %>%
    add_trace(x = ~group, y = ~Percentage, type = "scatter", mode = "linesx", alpha = 0.8,
              color = I(lmn_colors[2]), name = "Percentage xx", yaxis = "y2") %>%
    add_trace(x = ~group, y = ~average, type = "scatter", mode = "linesx", alpha = 0.4,
              color = I(lmn_colors[5]), name = "Avg percentage xx", yaxis = "y2") %>%
    layout(yaxis = list(side = "left", title = "Total"),
           yaxis2 = list(side = "right", overlaying = "y", title = "Percentage xx"),
           xaxis = list(title = paste0(variable, " (logarithmic scale)")))
}

# Filter only active members who do not have opt out profiling. (includes OPT_OUT)
# Fill NA with UNKOWN to make sure these are kept
# TODO: these filters only work on ATP_DBM_VIEW.membership table.
filter_active <- function(dataframe) {
  dataframe %>% 
    mutate(optout_profiling = ifelse(is.na(optout_profiling), "UNKNOWN", optout_profiling)) %>% 
    filter(msh_actief == 1 & optout_profiling != "NIET PROFILEREN" & msh_stduitsluiting == 0) %>% 
    filter(!z_msh %in% c(5056457, 5056458, 5056455, 5056456, 6640423, 7268459, 7475507)) # standaard uitsluitingen kaarten
}

# Filter only active members who are opt-in and do not have opt out profiling.
# Fill NA with UNKOWN to make sure these are kept
filter_active_optin <- function(dataframe) {
  dataframe %>% 
    mutate(optout_profiling = ifelse(is.na(optout_profiling), "UNKNOWN", optout_profiling)) %>% 
    filter(msh_actief == 1 & msh_optin == 1 & optout_profiling != "NIET PROFILEREN"  & msh_stduitsluiting == 0) %>% 
    filter(!z_msh %in% c(5056457, 5056458, 5056455, 5056456, 6640423, 7268459, 7475507)) # standaard uitsluitingen kaarten
}

# Filter only active members who are opt-in and do not have opt out profiling.
# Fill NA with UNKOWN to make sure these are kept
filter_active_optin_email <- function(dataframe) {
  dataframe %>% 
    mutate(optout_profiling = ifelse(is.na(optout_profiling), "UNKNOWN", optout_profiling)) %>% 
    filter(msh_actief == 1 & msh_emailbaar == 1 & msh_optin == 1 & optout_profiling != "NIET PROFILEREN"  & msh_stduitsluiting == 0) %>% 
    filter(!z_msh %in% c(5056457, 5056458, 5056455, 5056456, 6640423, 7268459, 7475507)) # standaard uitsluitingen kaarten
}

# Create age from birthday using several lubridate functions
feature_age <- function(dataframe, variable) {
  dataframe %>% 
    mutate(age = floor(interval(.data[[variable]], today()) / duration(n = 1, "years")))
}

# Create participation years from start date
feature_participation <- function(dataframe, variable) {
  dataframe %>% 
    mutate(participation = floor(interval(.data[[variable]], today()) / duration(n = 1, "years")))
}

#Function for Corrplot with many variables. Only show variable with an absolute correlation of 0.5 or higher. 
# Known "error" does not allow Date variables, but also not correcting for them. 
corr_simple <- function(data = df, sig = 0.5) {
  #convert data to numeric in order to run correlations
  #convert to factor first to keep the integrity of the data - each value will become a number rather than turn into NA
  df_cor <- data %>% mutate_if(is.character, as.factor)
  df_cor <- df_cor %>% mutate_if(is.factor, as.numeric)
  #run a correlation and drop the insignificant ones
  corr <- cor(df_cor)
  #prepare to drop duplicates and correlations of 1     
  corr[lower.tri(corr, diag = TRUE)] <- NA 
  #drop perfect correlations
  corr[corr == 1] <- NA 
  #turn into a 3-column table
  corr <- as.data.frame(as.table(corr))
  #remove the NA values from above 
  corr <- na.omit(corr) 
  #select significant values  
  corr <- subset(corr, abs(Freq) > sig) 
  #sort by highest correlation
  corr <- corr[order(-abs(corr$Freq)), ] 
  #print table
  print(corr)
  #turn corr back into matrix in order to plot with corrplot
  mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var = "Freq")
  #plot correlations visually
  corrplot(mtx_corr, is.corr = FALSE, tl.col = "black", na.label = " ", method = "color", type = "upper", addCoef.col = "black")
}


corr_cat <- function(data = df, sig = 0.5) {
  #convert char to factor in order to run correlations
  df_cor <- data %>% select_if(is.character)
  
  
}


#Functions for detection and removal of outliers
# define iqr
outliers <- function(x) {
  lower <- quantile(x, probs = .001)
  upper <- quantile(x, probs = .999)
  x > upper | x < lower
}
#remove rows
remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]), ]
  }
  df
}


# TODO: evaluate whether functions below are relevant for us.

# Get standard coefficients (can be used to compare estimates on a more standard
# way), and build a plot of the estimates with error bars for confidence
# intervals and p-value gets a color. Numbers in brackets are the total number
# of observations in a category.
get_std_coef <- function(fit) {
  modelmat <- model.matrix(fit$formula, fit$data)
  std_x <- apply(modelmat, 2, function(x) pi*sqrt(3)/sd(x))
  counts <- apply(modelmat, 2, sum)
  out <- tibble(var = names(std_x), std_x) %>% 
    left_join(tibble(var = names(coef(fit)), original_coef = coef(fit)), by = "var") %>% 
    mutate(std_coef = std_x * original_coef) %>% 
    left_join(tibble(var = names(counts), counts), by = "var") %>% 
    mutate(not_factor = var %in% attr(terms(fit), "term.labels")) %>% 
    mutate(counts = if_else(not_factor, NA_integer_, as.integer(counts))) %>% 
    filter(var != "(Intercept)",) %>% 
    select(var, original_coef, std_coef, counts)
  out
}

plot_fit_summary <- function(fit, ylim = NULL) {
  tidy_fit_summary(fit) %>%
    ggplot() + geom_point(aes(x = category_count, y = exp(estimate), color = p.value)) + 
    geom_hline(aes(yintercept = 1)) +
    geom_errorbar(aes(x = category_count, ymin = exp(estimate - 1.96*std.error), ymax = exp(estimate + 1.96*std.error))) +
    scale_color_gradient2(midpoint = 0.05, low = "darkgreen", mid = "orange", high = "darkred") +
    facet_grid(variable~., scales = "free_y", space = "free_y", switch = "y") + 
    coord_flip(ylim = ylim) + theme_minimal() +
    theme(strip.text.y = element_text(angle = 180),
          strip.placement = "outside")
}

tidy_fit_summary <- function(fit) {
  tidy(fit) %>% 
    left_join(get_std_coef(fit), by = c("term" = "var")) %>% 
    mutate(variable = str_replace(term, paste0("(", paste0(attr(terms(fit), "term.labels"), collapse = "|"), ")(.*)"), "\\1"), 
           category = str_replace(term, paste0("(", paste0(attr(terms(fit), "term.labels"), collapse = "|"), ")(.*)"), "\\2")) %>% 
    mutate(category_count = ifelse(!is.na(counts), 
                                   paste0(category, " (", counts, ")"),
                                   "")) %>% 
    mutate(exp_coef = exp(estimate),
           stderr_min = exp(estimate) - exp(estimate - 1.96 * std.error),
           stderr_max = exp(estimate + 1.96 * std.error) - exp(estimate))
}

plotly_fit_summary <- function(fit) {
  category_plotlys <- tidy_fit_summary(fit) %>% 
    mutate(variable_name = variable,
           category_count = if_else(category_count == "", variable_name, category_count)) %>% 
    group_by(variable) %>% 
    group_map(~plot_ly(data = ., y = ~category_count, x = ~exp(estimate), 
                       type = "scatter", mode = "markers", 
                       name = ~variable_name, error_x = ~list(array = stderr_max, arrayminus = stderr_min)) %>%
                layout(shapes = list(vline(1))))
  
  subplot_info <- tidy_fit_summary(fit) %>% 
    mutate(variable_name = variable,
           category_count = if_else(category_count == "", variable_name, category_count)) %>% 
    count(variable) %>% 
    mutate(heights = n/sum(n))
  
  subplot(category_plotlys, nrows = nrow(subplot_info), shareX = TRUE, heights = subplot_info$heights)
}

vline <- function(x = 0, color = "black") {
  list(
    type = "line", 
    y0 = 0, 
    y1 = 1, 
    yref = "paper",
    x0 = x, 
    x1 = x, 
    line = list(color = color)
  )
}


# gets data from atp_dbm_view from transactietable
# takes a list of partners as input
# takes a list of transactieredenen as input
# takes 3 years of history
# sel_partners        <- c("Praxis Online", "Shell", "AHonline", "Albert Heijn", "Praxis")
# sel_transactiereden <- c("ZSPAREN", "ZAMBONUS", "ZBONUS", "ZINWISSELEN")

create_df_tx <- function(partners_list, transactiereden_list) {
  
  maxtxdat <- tbl(con, in_schema("atp_dbm_view", "transactie")) %>% 
    dplyr::summarise(maxtransactiedatum = max(transactiedatum, na.rm = TRUE)) %>% 
    as.data.frame()
  maxtxdat_ry1 <- maxtxdat[1, 1] - years(1)
  maxtxdat_ry2 <- maxtxdat[1, 1] - years(2)
  maxtxdat_ry3 <- maxtxdat[1, 1] - years(3)
  
  recency <- tbl(con, in_schema("atp_dbm_view", "transactie")) %>% 
    filter(transactiedatum > maxtxdat_ry3, transactiereden %in% transactiereden_list) %>% 
    select(membershipid, transactiedatum, transactiereden) %>% 
    group_by(membershipid, transactiereden) %>% 
    summarise(rectxdat = max(transactiedatum, na.rm = TRUE)) %>% 
    as.data.frame()
  
  setDT(recency)
  recwide <- dcast(recency, membershipid ~ transactiereden, value.var = c("rectxdat"))
  
  tx <- tbl(con, in_schema("atp_dbm_view", "transactie")) %>% 
    filter(transactiedatum > maxtxdat_ry3, transactiereden %in% transactiereden_list) %>% 
    select(membershipid, transactiedatum, transactiereden, partnernaam, punten) %>% 
    mutate(txd_ry = case_when(transactiedatum > maxtxdat_ry1 ~ "ry1",
                              transactiedatum > maxtxdat_ry2 ~ "ry2",
                              TRUE ~"ry3"),
           partner = ifelse(partnernaam %in% partners_list, partnernaam, "Other"),
           transactiereden = ifelse(transactiereden %in% c("ZAMBONUS", "ZBONUS"), "BONUS", transactiereden)) %>% 
    group_by(membershipid, txd_ry, partner, transactiereden) %>% 
    summarise(n_tx      = count(0),
              n_am      = sum(punten, na.rm = TRUE),
              mean_am   = mean(punten, na.rm = TRUE)
    ) %>% 
    ungroup() %>% 
    select(membershipid, txd_ry, partner, transactiereden, n_tx, n_am, mean_am) %>% 
    # pivot_wider(names_from = c(txd_ry, partner, transactiereden),
    #             values_from = c(n_tx, n_am, mean_am),
    #             values_fill = 0) %>%
    as.data.frame()
  
  setDT(tx)
  txwide <- dcast(tx, membershipid ~ txd_ry + partner + transactiereden, value.var = c("n_tx", "n_am", "mean_am"))
  
  tx_dat <- txwide %>% 
    left_join(recwide, by = c("membershipid"))
  
  return(tx_dat)
}

# Selection of base variables for membership, EDM and BAG for most ML models (Nicole/Joel)
select_base_vars <- function(dataframe) {
  dataframe %>% 
    select(
      mem_z_msh
      ,mem_z_mshtype
      ,mem_z_strtdat
      ,mem_birthday
      ,mem_gender 
      ,mem_z_bppcnum
      ,mem_z_bpadlcd 
      ,mem_city_1 
      ,mem_z_pntblnc
      # ,mem_msh_actief # droppen na filter
      ,mem_msh_emailbaar 
      ,mem_msh_optin
      ,mem_msh_postbaar
      ,mem_msh_stduitsluiting
      ,mem_lastlogon 
      ,mem_creditcard 
      ,mem_creditcardchdt 
      # ,mem_optout_profiling # droppen after filter
      # ,mem_tel_changedate # veel NA
      ,mem_shll_cpl 
      ,mem_cpl_changed_at
      ,mem_vakantieregio
      # ,edm_X_COORD_RD # we keep long/latitude
      # ,edm_Y_COORD_RD # we keep long/latitude
      ,edm_LONGITUDE
      ,edm_LATITUDE 
      ,edm_PROVINCIE
      ,edm_URB 
      ,edm_LEVENSFASE 
      ,edm_AANTAL_PERSONEN 
      ,edm_LFT_OUDSTE_KIND 
      ,edm_INKOMEN
      ,edm_KOOPKRACHT 
      ,edm_OPLEIDING 
      ,edm_SOCIALEKLASSE
      ,edm_KOSTW_FUNCTIE 
      ,edm_TWEEVERDIENER
      ,edm_WERKUREN
      ,edm_WONINGTYPE
      ,edm_WOONDOEL 
      ,edm_EIGENDOM 
      ,edm_WOZWAARDE 
      ,edm_WOZWAARDEONTWIKKELING
      ,edm_HUURPRIJSWONING 
      ,edm_WOONLASTEN 
      ,edm_BOUWJAAR 
      ,edm_OPPERVLAKTE 
      ,edm_WONINGINHOUD
      ,edm_VRIJERUIMTEPERCEEL 
      ,edm_AANTAL 
      ,edm_EERSTE_EIGENAAR
      ,edm_SEGMENT
      ,edm_PRIJS
      ,edm_LEEFTIJD_AUTO
      ,edm_ZAKELIJKE_AUTO
      ,edm_GEBRUIK_OV 
      ,edm_match
      ,bag_woonfunctie 
      ,bag_pand_bouwjaar
      ,bag_pand_oppervlakte
      ,bag_verblijfsobject_woningtype
      # ,bag_perceel_oppervlakte # veel NAs, edm versie lijkt beter
      # ,bag_perceel_oppervlakte_onbebouwd # veel NAs, edm versie lijkt beter
      ,bag_koophuur
      ,bag_match
    )
}

# TODO: dubbel check of er nog meer relevante BAG variabelen zijn?
select_bag_vars <- function(dataframe) {
  dataframe %>% 
    select(HH_Sleutel,
           identificatie,
           huisnummer,
           huisnummer_toevoeging,
           woonfunctie 
           ,pand_bouwjaar
           ,pand_oppervlakte
           ,verblijfsobject_woningtype
           # ,perceel_oppervlakte # veel NAs, edm versie lijkt beter
           # ,perceel_oppervlakte_onbebouwd # veel NAs, edm versie lijkt beter
           ,koophuur)
}
