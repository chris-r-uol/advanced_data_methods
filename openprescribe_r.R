library(httr2)
library(readr)
library(dplyr)
library(reticulate)
library(sf)
library(jsonlite)

codes <- c(
  lincolnshire = "71E",
  leeds = "15F",
  bradford = "36J",
  cheshire = "27D",
  nottingham = "52R",
  portsmouth = "10R",
  southampton = "D9Y0V",
  banes = "92G",
  bristol = "15C",
  newcastle = "13T",
  basildon = "99E",
  bwv = "D4U1Y",
  coventry = "B2M3M",
  solihul = "15E",
  birmingham = "15E",
  leicester = "04C",
  derby = "15M",
  sheffield = "03N",
  rotherham = "03L",
  fareham = "D9Y0V",
  tyneside = "13T"
)

# We need to create a python virtual environment to run the cloudscraper package

virtualenv_create("openprescribing-r")

virtualenv_install(
  envname = "openprescribing-r",
  packages = "cloudscraper",
  pip = TRUE
)

py_module_available("cloudscraper")


get_openprescribe_data <- function(la_code, measure, cleaned = TRUE) {
  # Example Usage
  
  #df <- get_openprescribe_data(
  #  la_code = "leeds",
  #  measure = "saba"
  #)
  
  la_code <- tolower(la_code)
  
  if (!la_code %in% names(codes)) {
    stop(
      "Unknown la_code: ", la_code,
      ". Valid options are: ",
      paste(names(codes), collapse = ", ")
    )
  }
  
  code <- codes[[la_code]]
  
  url <- paste0(
    "https://openprescribing.net/api/1.0/measure_by_practice/",
    "?format=csv",
    "&org=", code,
    "&parent_org_type=ccg",
    "&measure=", measure
  )
  
  cloudscraper <- import("cloudscraper")
  
  scraper <- cloudscraper$create_scraper(
    browser = dict(
      browser = "chrome",
      platform = "windows",
      desktop = TRUE
    )
  )
  
  response <- scraper$get(url)
  
  if (response$status_code != 200) {
    stop(
      "Error fetching OpenPrescribing data. HTTP status: ",
      response$status_code
    )
  }
  
  csv_text <- response$text
  
  df <- read_csv(
    I(csv_text),
    show_col_types = FALSE
  )
  
  if (cleaned) {
    orgs_with_none <- df |>
      filter(is.na(calc_value)) |>
      pull(org_id) |>
      unique()
    
    df <- df |>
      filter(!org_id %in% orgs_with_none) |>
      rename(site = org_name)
  }
  
  df
}

get_openprescribe_metadata <- function(la_code) {
  

  la_code <- tolower(la_code)
  
  if (!la_code %in% names(codes)) {
    stop(
      "Unknown la_code: ", la_code,
      ". Valid options are: ",
      paste(names(codes), collapse = ", ")
    )
  }
  
  code <- codes[[la_code]]
  
  url <- paste0(
    "https://openprescribing.net/api/1.0/org_location/",
    "?q=", code
  )
  
  cloudscraper <- import("cloudscraper")
  
  scraper <- cloudscraper$create_scraper(
    browser = dict(
      browser = "chrome",
      platform = "windows",
      desktop = TRUE
    )
  )
  
  response <- scraper$get(url)
  
  if (response$status_code != 200) {
    stop(
      "Error fetching OpenPrescribing metadata. HTTP status: ",
      response$status_code
    )
  }
  
  json_text <- response$text
  
  # Convert GeoJSON text to an sf object
  gdf <- st_read(
    json_text,
    quiet = TRUE
  )
  
  # Add longitude and latitude based on geometry centroids
  centroids <- st_centroid(st_geometry(gdf))
  
  coords <- st_coordinates(centroids)
  
  gdf <- gdf |>
    mutate(
      longitude = coords[, "X"],
      latitude = coords[, "Y"]
    ) |>
    rename(site = name)
  
  gdf
}


metadata <- get_openprescribe_metadata("leeds")
