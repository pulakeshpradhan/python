// -------------------------------------------------------------------------------------
//                              Import Rquired Datasets
// -------------------------------------------------------------------------------------
var elevation = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Elevation"),
    dist_to_river = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Distance_to_River"),
    drainage_density = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Drainage_Density"),
    geomorphology = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Geomorphology"),
    lithology = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Lithology"),
    mfi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/MFI"),
    mndwi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/MNDWI"),
    ndvi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/NDVI"),
    rainfall = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Rainfall"),
    relief_amplitude = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Relief_Amplitude"),
    spi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/SPI"),
    sti = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/STI"),
    slope = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/Slope"),
    tpi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/TPI"),
    tri = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/TRI"),
    twi = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/TWI"),
    flood_points = ee.FeatureCollection("projects/research-projects-385305/assets/Flood/Maldah/Maldah_Flood_Points"),
    non_flood_points = ee.FeatureCollection("projects/research-projects-385305/assets/Flood/Maldah/Maldah_Non_Flood_Points"),
    lulc = ee.Image("projects/research-projects-385305/assets/Flood/Maldah/LULC"),
    clay_content = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02");

// -------------------------------------------------------------------------------------
//                        Import the Region of Interest
// -------------------------------------------------------------------------------------

var roi = ee.FeatureCollection("users/geonextgis/Maldah_Projected");
var roi_style = {
  color: "red",
  fillColor: "00000000",
  width: 1.5
};
Map.addLayer(roi.style(roi_style), {}, "Region of Interest");
Map.centerObject(roi, 10);

// -------------------------------------------------------------------------------------
//                             Prepare the Parameters
// -------------------------------------------------------------------------------------

// Prepare the clay content parameter
var clay_content = clay_content.select("b200")
                               .clip(roi);

// Resample the layer into 30 meters
var clay_content = clay_content.resample("bilinear")
                              .reproject({crs:"EPSG:32645", scale:30})
                              .toDouble()
                              .clip(roi);

// Change the band names of the parameters
var elevation = elevation.rename("Elevation")
                         .where(elevation.eq(65535), 5);
var slope = slope.rename("Slope");
var dist_to_river = dist_to_river.rename("Dist_to_River");
var drainage_density = drainage_density.rename("Drainage_Density");
var geomorphology = geomorphology.rename("Geomorphology");
var lithology = lithology.rename("Lithology");
var relief_amplitude = relief_amplitude.rename("Relief_Amplitude");
var rainfall = rainfall.rename("Rainfall");
var mfi = mfi.rename("MFI");
var ndvi = ndvi.rename("NDVI");
var mndwi = mndwi.rename("MNDWI");
var spi = spi.rename("SPI");
var sti = sti.rename("STI");
var tpi = tpi.rename("TPI");
var tri = tri.rename("TRI");
var twi = twi.rename("TWI");
var lulc = lulc.rename("LULC");
var clay_content = clay_content.rename("Clay_Content").unmask(0);

// Store all the band names in a list
var bandNames = ["Elevation", "Slope", "Dist_to_River", "Drainage_Density",
                 "Geomorphology", "Lithology", "Relief_Amplitude", "Rainfall",
                 "MFI", "NDVI", "MNDWI", "SPI",
                 "STI", "TPI", "TRI", "TWI", "LULC", "Clay_Content"];
                 
// Store the numerical and categorical variables in separate list
var numericalBandNames = ["Elevation", "Slope", "Dist_to_River", "Drainage_Density",
                          "Relief_Amplitude", "Rainfall", "MFI", "NDVI", "MNDWI", 
                          "SPI", "STI", "TPI", "TRI", "TWI", "Clay_Content"];
                 
var categoricalBandNames = ["Geomorphology", "Lithology", "LULC"];

// Store the image sequence
var imgSequence = [elevation, slope, dist_to_river, drainage_density,
                   geomorphology, lithology, relief_amplitude, rainfall,
                   mfi, ndvi, mndwi, spi,
                   sti, tpi, tri, twi, lulc, clay_content];
                   

// Create a single parameter image
var parameters = ee.Image(imgSequence)
                   .select(bandNames)
                   .clip(roi)
                   .toDouble();
                   
print("Parameters Image:", parameters);

// -------------------------------------------------------------------------------------
//                              Normalize the Image
// -------------------------------------------------------------------------------------

// Create a function to Normalize the image
function normalize(image){
  
  // Define the numerical and categorical bands
  var numericalBands = image.select(numericalBandNames);
  var categoricalBands = image.select(categoricalBandNames);
  
  // Extract minimum values of the numerical bands
  var minDict = numericalBands.reduceRegion({
    reducer: ee.Reducer.min(), 
    geometry: roi,
    scale: 30,
    bestEffort: true,
    maxPixels: 1e10, 
    tileScale: 16
  });
  
  // Extract maximum values of the numerical bands
  var maxDict = numericalBands.reduceRegion({
    reducer: ee.Reducer.max(),
    geometry: roi,
    scale: 30,
    bestEffort: true,
    maxPixels: 1e10, 
    tileScale: 16
  });
  
  // Create constant images with min and max values
  var mins = ee.Image.constant(minDict.values(numericalBandNames));
  var maxs = ee.Image.constant(maxDict.values(numericalBandNames));
  
  // Normalize the numerical bands
  // Normalize = (X - Xmin) / (Xmax - Xmin)
  var norm_numerical_bands = numericalBands.subtract(mins)
                                           .divide(maxs.subtract(mins));
                                           
  // Add the categorical bands
  var normalized_image = ee.Image([norm_numerical_bands, categoricalBands]);
  return normalized_image.select(bandNames);
}

// Apply the normalize function over the parameter image
var normalized = normalize(parameters);
print("Normalized Image:", normalized);

// Display the Normalized NDVI band
var norm_ndvi = normalized.select("NDVI");
var ndviVis = {
  min: 0,
  max: 1,
  palette: ['#ffffe5','#f7fcb9','#d9f0a3','#addd8e',
            '#78c679','#41ab5d','#238443','#005a32']
};
Map.addLayer(norm_ndvi, ndviVis, "Normalized NDVI");

// -------------------------------------------------------------------------------------
//                               Prepare the Sample Data
// -------------------------------------------------------------------------------------

// Set flood attribute to 1 and 0 for flood and non flood points respectively
var flood_points = flood_points.map(function (f){
  return f.set({"Flood": 1});
}).select("Flood");

var non_flood_points = non_flood_points.map(function (f){
  return f.set({"Flood": 0});
}).select("Flood");

// Merge the flood and non flood points
var sample_points = flood_points.merge(non_flood_points);
print("Total Sample Points (Flood and Non Flood)", sample_points.size());

// -------------------------------------------------------------------------------------
//                               Collect the sample data
// -------------------------------------------------------------------------------------

// Collect the band values from Normalized image
var sample_data = normalized.sampleRegions({
  collection: sample_points, 
  properties: ["Flood"],
  scale: 30, 
  projection: "EPSG:32645",
  tileScale: 16,
  geometries: true
});
print("First Element of the Sample Data:", sample_data.first());

// -------------------------------------------------------------------------------------
//                   Export the Normalized Image and Sample Data
// -------------------------------------------------------------------------------------

// Export the Normalized Image
Export.image.toDrive({
  image: normalized.toDouble(),
  description: "Maldah_Flood_Parameters",
  folder: "GEE", 
  fileNamePrefix: "Maldah_Flood_Parameters",
  region: roi,
  scale: 30, 
  crs: "EPSG:32645",
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

// Export the Sample Data
Export.table.toDrive({
  collection: sample_data,
  description: "Flood_Sample_Data",
  folder: "GEE", 
  fileNamePrefix: "Flood_Sample_Data", 
  fileFormat: "SHP"
});

