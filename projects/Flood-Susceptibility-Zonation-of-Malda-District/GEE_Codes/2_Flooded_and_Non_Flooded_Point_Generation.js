// --------------------------------------------------------------------------------
//                         Import Required Datasets
// --------------------------------------------------------------------------------
var roi = ee.FeatureCollection("users/geonextgis/Maldah_Projected"),
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater"),
    srtm = ee.Image("USGS/SRTMGL1_003");

// --------------------------------------------------------------------------------
//                         Import the Region of Interest
// --------------------------------------------------------------------------------

// Display the Region of Interest
var roi_style = {
    color: "red",
    fillColor: "00000000",
    width: 1.5
  };
  Map.addLayer(roi.style(roi_style), {}, "Region of Interest");
  Map.centerObject(roi, 10);
  
  // --------------------------------------------------------------------------------
  //                          Function to Apply Speckle Filter
  // --------------------------------------------------------------------------------
  
  // Create a function to convert from dB to Natural
  function toNatural(image){
    return ee.Image(10.0).pow(image.select(0).divide(10.0));
  }
  
  // Create a function to convert from Natural to dB
  function todB(image){
    return ee.Image(image).log10().multiply(10.0);
  }
  
  // Apply a Refined Lee Speckle filter as coded in the SNAP 3.0 S1TBX:
  // Adapted by Guido Lemoine
  
  function RefinedLee(image){
    // Image must be in natural units, i.e., not in dB.
    // Set up a 3x3 kernels
    var weights3 = ee.List.repeat(ee.List.repeat({value: 1, count: 3}), 3);
    var kernel3 = ee.Kernel.fixed({width: 3, 
                                   height:3, 
                                   weights: weights3,
                                   x: 1,
                                   y: 1, 
                                   normalize: false});
    
    var mean3 = image.reduceNeighborhood({reducer: ee.Reducer.mean(),
                                          kernel: kernel3});
    var variance3 = image.reduceNeighborhood({reducer: ee.Reducer.variance(),
                                              kernel: kernel3});     
                                              
    // Use a sample of the 3x3 windows inside a 7x7 window sto determine gradients
    // and directions
    var sample_weights = ee.List([[0, 0, 0, 0, 0, 0, 0], 
                                  [0, 1, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0], 
                                  [0, 1, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]);
                                  
    var sample_kernel = ee.Kernel.fixed({width: 7, 
                                         height: 7,
                                         weights: sample_weights, 
                                         x: 3, 
                                         y: 3, 
                                         normalize: false});
    
    // Calculate the mean and variance for the sampled windows and store as 9 bands
    var sample_mean = mean3.neighborhoodToBands(sample_kernel);
    var sample_var = variance3.neighborhoodToBands(sample_kernel);
    
    // Determine the 4 gradients for the sampled windows
    var gradients = sample_mean.select(1).subtract(sample_mean.select(7).abs());
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
    
    // And find the maximum gradient amongst gradient bands
    var max_gradient = gradients.reduce(ee.Reducer.max());
    
    // Create a mask for band pixels that are the maximum gradient
    var gradmask = gradients.eq(max_gradient);
    
    // Duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask);
    
    // Determine the 8 directions
    var directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4)
                                .subtract(sample_mean.select(7))).multiply(1);
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4))
                           .gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4))
                           .gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4))
                           .gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
    
    // The next 4 are the not() of the previous 4
    directions = directions.addBands(directions.select(0).not().multiply(5));
    directions = directions.addBands(directions.select(1).not().multiply(6));
    directions = directions.addBands(directions.select(2).not().multiply(7));
    directions = directions.addBands(directions.select(3).not().multiply(8));
    
    // Mask all values that are not 1-8
    directions = directions.updateMask(gradmask);
    
    // "collapse" the stack into a single band image (due to masking, each pixel has
    // just one value (1-8) in it's directional band, and is otherwise masked)
    directions = directions.reduce(ee.Reducer.sum());
    
    // var pal = ["ffffff", "ff0000", "ffff00", "00ff00", "00ffff", "0000ff", "ff00ff", "000000"];
    // Map.addLayer(directions.reduce(ee.Reducer.sum()), {min: 1, max: 8, palette: pal}, "Directions", false)
    
    var sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
    
    // Calculate localNoiseVariance
    var sigmaV = sample_stats.toArray().arraySort().arraySlice(0, 0, 5)
                             .arrayReduce(ee.Reducer.mean(), [0]);
                             
    // Set up the 7*7 kernels for directional statistics
    var rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3)
                         .cat(ee.List.repeat(ee.List.repeat(1, 7), 4));
                         
    var diag_weights = ee.List([[1, 0, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1]]);
                                
    var rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, false);
    var diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, false);
    
    // Create stacks for mean and variance using the original kernels. Mask with 
    // relevant direction.
    var dir_mean = image.reduceNeighborhood(ee.Reducer.mean(), rect_kernel)
                        .updateMask(directions.eq(1));
    var dir_var = image.reduceNeighborhood(ee.Reducer.variance(), rect_kernel)
                       .updateMask(directions.eq(1));
    dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(), diag_kernel)
                        .updateMask(directions.eq(2)));
    dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(), diag_kernel)
                        .updateMask(directions.eq(2)));
    
    // Add the bands for rotated kernels
    for (var i=1; i<4; i++){
      dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i))
                         .updateMask(directions.eq(2*i+1)));
      dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i))
                         .updateMask(directions.eq(2*i+1)));                       
      dir_mean = dir_mean.addBands(image.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i))
                         .updateMask(directions.eq(2*i+2)));                       
      dir_var = dir_var.addBands(image.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i))
                         .updateMask(directions.eq(2*i+2)));                            
    }
    
    // "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional
    //   band, and is otherwise masked)
    dir_mean = dir_mean.reduce(ee.Reducer.sum());
    dir_var = dir_var.reduce(ee.Reducer.sum());
    
    // A finally generate the filtered value
    var varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV))
                      .divide(sigmaV.add(1.0));
    var b = varX.divide(dir_var);
    
    var result = dir_mean.add(b.multiply(image.subtract(dir_mean)));
    
    return (result.arrayFlatten([["sum"]]));
  }
  
  // --------------------------------------------------------------------------------
  //                         Function to Generate Flood Patches 
  // --------------------------------------------------------------------------------
  
  function generateFloodPatches(date){
    
    // Select images by predefined dates
    var eventDate = ee.Date(date);
    var beforeStart = eventDate.advance(-15, "day");
    var afterEnd = eventDate.advance(15, "day");
    
    // Import the Sentinel-1 SAR GRD image collection
    var s1 = ee.ImageCollection("COPERNICUS/S1_GRD");
  
    // Filter the image collection
    var s1Filtered = s1.filterBounds(roi)
                      .filter(ee.Filter.eq("instrumentMode", "IW"))
                      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                      .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                      .filter(ee.Filter.eq("resolution_meters", 10))
                      .select("VH");
    
    // Before Image Acquistion Date
    var beforeImage = s1Filtered.filterDate(beforeStart, eventDate)
                                .sort("system:time_start", false)
                                .first()
                                .clip(roi);
  
    // After Image Acquistion Date                          
    var afterImage = s1Filtered.filterDate(eventDate, afterEnd)
                               .sort("system:time_start", false)
                               .first()
                               .clip(roi);
                               
    // Apply the Speckle filter on the before and after image
    var beforeFiltered = ee.Image(todB(RefinedLee(toNatural(beforeImage.select("VH")))));
    var afterFiltered = ee.Image(todB(RefinedLee(toNatural(afterImage.select("VH")))));
    
    // Create a difference image
    var difference = afterFiltered.divide(beforeFiltered);
  
    // Define a threshold
    var diffThreshold = 1.25;
    
    // Initial estimate of flooded pixels
    var flooded = difference.gt(diffThreshold).rename("water").selfMask();
    
    // Mask out area with permanent/semi-permanent water
    var permanentWater = gsw.select("seasonality").gte(5).clip(roi);
    
    // GSW data is masked in non-water areas. Set it to 0 using unmask().
    // Invert the image to set all non-permanent region to 1
    var permanentWaterMask = permanentWater.unmask(0).not();
    
    var flooded2 = flooded.updateMask(permanentWaterMask);
    
    // Mask out areas with more than 5 degree slope using the SRTM DEM
    var slopeThreshold = 5;
    var slope = ee.Terrain.slope(srtm.clip(roi));
    var steepAreas = slope.gt(slopeThreshold);
    var slopeMask = steepAreas.not();
    
    var flooded3 = flooded2.updateMask(slopeMask);
    
    // connectedPixelCount is zoom dependent, so visual result will vary
    var connectedPixelThreshold = 20;
    var connections = flooded.connectedPixelCount(25);
    var disconnectedAreas = connections.lt(connectedPixelThreshold);
    var disconnectedAreaMask = disconnectedAreas.not();
    
    var flooded4 = flooded3.updateMask(disconnectedAreaMask);
  
    return flooded4.rename("Flooded");
  }
  
  // --------------------------------------------------------------------------------
  //                    Generate Flood Patches for Past Events
  // --------------------------------------------------------------------------------
  
  var flood2021 = generateFloodPatches("2021-08-10");
  Map.addLayer(flood2021, {min:0, max:1, palette: ["red"]}, "Flood Patches, 2021", false);
  
  var flood2019 = generateFloodPatches("2019-09-25");
  Map.addLayer(flood2019, {min:0, max:1, palette: ["red"]}, "Flood Patches, 2019", false);
  
  var flood2017 = generateFloodPatches("2017-08-17");
  Map.addLayer(flood2017, {min:0, max:1, palette: ["red"]}, "Flood Patches, 2017", false);
  
  // Merge all the flood patches
  var floodedColl = ee.ImageCollection.fromImages([flood2017, flood2019, flood2021]);
  var floodPatches = floodedColl.reduce(ee.Reducer.anyNonZero());
  Map.addLayer(floodPatches, {min:0, max:1, palette: ["blue"]}, "Flood Patches", false);
  
  // --------------------------------------------------------------------------------
  //                         Apply Morphological Operations
  // --------------------------------------------------------------------------------
  
  // Do unmask() to replace masked pixels with 0
  // This avoids extra pixels along the edges
  var floodPatches = floodPatches.unmask();
  
  // Perform a morphological closing to fill holes in flooded patches
  var waterProcessed = floodPatches.focalMax({
    radius: 30,
    kernelType: "square",
    units: "meters", 
    iterations: 1
    }).focalMin({
      radius: 30,
      kernelType: "square", 
      units: "meters", 
      iterations: 1
    }).selfMask().rename("Flood").clip(roi);
    
  Map.addLayer(waterProcessed, {min:0, max:1, palette: ["blue"]}, "Flood Patches (Processed)", false);
  
  // --------------------------------------------------------------------------------
  //                        Generate Random Flood Sample Points
  // --------------------------------------------------------------------------------
  
  var flood_samples = waterProcessed.sample({
    region: roi.geometry(),
    scale: 30,
    numPixels: 5000, 
    seed: 0,
    dropNulls: true, 
    tileScale: 16,
    geometries: true
  });
  
  var flood_point_style = {
    color: "black", 
    pointSize: 2,
    pointShape: "circle", 
    width: 0.5, 
    fillColor: "#d73027"
  };
  // Map.addLayer(flood_samples.style(flood_point_style), {color:"red"}, "Flood Points", false);
  print("Flood Points:", flood_samples);
  
  // Export the Flood Points
  Export.table.toDrive({
    collection: flood_samples,
    description: "Maldah_Flood_Points", 
    folder: "GEE", 
    fileNamePrefix: "Maldah_Flood_Points", 
    fileFormat: "SHP"
  });
  
  // --------------------------------------------------------------------------------
  //                      Generate Random Non-Flood Sample Points
  // --------------------------------------------------------------------------------
  
  // Display DEM of the ROI
  var dem = srtm.clip(roi);
  var dem_palette = ['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850'];
  dem_palette.reverse();
  var dem_vis = {
    min: 10,
    max: 50,
    palette: dem_palette
  };
  Map.addLayer(dem, dem_vis, "ROI DEM");
  
  // Convert the flood patches into vector
  var floodPatchVec = waterProcessed.reduceToVectors({
    geometry: roi,
    scale: 100,
    eightConnected: false,
    bestEffort: true,
    maxPixels: 1e10,
    tileScale: 16
  });
  
  var floodPatchStyle = {
    color: "#2166ac",
    fillColor: "#4393c3",
    width: 1
  };
  Map.addLayer(floodPatchVec.style(floodPatchStyle), {}, "Flood Patches Vector");
  
  // Create a buffer of 250m around the flood patches
  var floodBuffer = floodPatchVec.geometry().buffer(500);
  var floodBufferStyle = {
    color: "#b2182b",
    fillColor: "#d6604d",
    width: 1
  };
  Map.addLayer(floodBuffer, {color: "#b2182b", fillColor: "#d6604d", width: 1}, "Flood Buffer", false);
  
  // Convert the flood buffer vector into a raster
  var floodBufferRas = ee.Image(0).paint(floodBuffer, 1).clip(roi);
  Map.addLayer(floodBufferRas, {min: 0, max:1}, "Flood Buffer Raster", false);
  
  // Mask out area with permanent/semi-permanent water
  var permanentWater = gsw.select("seasonality").gte(5).clip(roi);
    
  // GSW data is masked in non-water areas. Set it to 0 using unmask().
  // Invert the image to set all non-permanent region to 1
  var permanentWaterMask = permanentWater.unmask(0).not();
  
  // Select the area with 0 values
  var nonflood = floodBufferRas.not()
                               .updateMask(permanentWaterMask)
                               .updateMask(waterProcessed.unmask().eq(0))
                               .selfMask()
                               .rename("Non_Flood")
                               .clip(roi);
  Map.addLayer(nonflood, {min: 0, max:1}, "Non Flood Area", false);
  
  // Generate random non-flood samples
  var non_flood_samples = nonflood.sample({
    region: roi.geometry(),
    scale: 30,
    numPixels: 2000, 
    seed: 0,
    dropNulls: true, 
    tileScale: 16,
    geometries: true
  });
  
  var non_flood_point_style = {
    color: "black", 
    pointSize: 2,
    pointShape: "circle", 
    width: 0.5, 
    fillColor: "#1a9850"
  };
  print("Non Flood Points:", non_flood_samples);
  
  // Display the Flood and Non Flood Points
  Map.addLayer(flood_samples.style(flood_point_style), {color:"red"}, "Flood Points");
  Map.addLayer(non_flood_samples.style(non_flood_point_style), {}, "Non Flood Points");
  
  // Export the Non Flood Points
  Export.table.toDrive({
    collection: non_flood_samples,
    description: "Maldah_Non_Flood_Points", 
    folder: "GEE", 
    fileNamePrefix: "Maldah_Non_Flood_Points", 
    fileFormat: "SHP"
  });
  