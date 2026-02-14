// --------------------------------------------------------------------------------
//                           Import Required Datasets
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
  //                       Prepare Sentinel-1 SAR GRD Imagery
  // --------------------------------------------------------------------------------
  
  // On 12 August 2021, severe floods affected the Maldah district of West Bengal.
  // At least 30 villages in Malda district were inundated.
  
  // Select images by predefined dates
  var beforeStart = "2021-08-01";
  var beforeEnd = "2021-08-10";
  var afterStart = "2021-08-10";
  var afterEnd = "2021-08-22";
  
  // Import the Sentinel-1 SAR GRD image collection
  var s1 = ee.ImageCollection("COPERNICUS/S1_GRD");
  
  // Filter the image collection
  var s1Filtered = s1.filterBounds(roi)
                     .filter(ee.Filter.eq("instrumentMode", "IW"))
                     .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                     .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                     .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                     .filter(ee.Filter.eq("resolution_meters", 10))
                     .select(["VV", "VH"]);
  
  // Before Image Acquistion Date: 2021-08-09
  var beforeImage = s1Filtered.filterDate(beforeStart, beforeEnd)
                              .first()
                              .clip(roi);
  
  // After Image Acquistion Date: 2021-08-21                            
  var afterImage = s1Filtered.filterDate(afterStart, afterEnd)
                             .first()
                             .clip(roi);
  // print(beforeColl)
  // print(afterColl)
  // Map.addLayer(beforeColl)
  // Map.addLayer(afterColl)
  
  // Create a function to add ratio band
  function addRatioBand(image){
    var ratioBand = image.select("VV").divide(image.select("VH")).rename("VV/VH");
    return image.addBands(ratioBand);
  }
  
  var beforeImage = addRatioBand(beforeImage);
  var afterImage = addRatioBand(afterImage);
  
  // Define a visualization
  var visParams = {
    min: [-25, -25, 0],
    max: [0, 0, 2]
  };
  Map.addLayer(beforeImage, visParams, "Before Floods", false);
  Map.addLayer(afterImage, visParams, "After Floods", false);
  
  // Visualize the changes by creating a composite image
  // Band Combination: VH-before image, VH-after image, and VH-before image
  var compositeImage = ee.Image.cat([beforeImage.select("VH").rename("VH_Before"),
                                     afterImage.select("VH").rename("VH_After"),
                                     beforeImage.select("VH").rename("VH_Before_2")]);
  Map.addLayer(compositeImage, {min: -25, max: -8}, "Change Composite");
  
  // --------------------------------------------------------------------------------
  //                              Apply Speckle Filter
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
  
  // Apply the Speckle filter on the before and after image
  var beforeFiltered = ee.Image(todB(RefinedLee(toNatural(beforeImage.select("VH")))));
  var afterFiltered = ee.Image(todB(RefinedLee(toNatural(afterImage.select("VH")))));
  
  Map.addLayer(beforeFiltered, {min:-25, max:0}, "VH before Speckle Filter", false);
  Map.addLayer(afterFiltered, {min:-25, max:0}, "VH after Speckle Filter", false);
  
  // --------------------------------------------------------------------------------
  //                              Apply a Threshold 
  // --------------------------------------------------------------------------------
  
  var difference = afterFiltered.divide(beforeFiltered);
  
  // Define a threshold
  var diffThreshold = 1.25;
  
  // Initial estimate of flooded pixels
  var flooded = difference.gt(diffThreshold).rename("water").selfMask();
  Map.addLayer(flooded, {min:0, max:1, palette: ["orange"]}, "Initial Flood Area", false);
  var roi = ee.FeatureCollection("users/geonextgis/Maldah_Projected"),
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater"),
    srtm = ee.Image("USGS/SRTMGL1_003");
  
  
  
  // --------------------------------------------------------------------------------
  //                                  Apply Masks 
  // --------------------------------------------------------------------------------
  
  // Mask out area with permanent/semi-permanent water
  var permanentWater = gsw.select("seasonality").gte(5).clip(roi);
  Map.addLayer(permanentWater.selfMask(), {min:0, max:1, palette: ["blue"]}, "Permanent Water", false);
  
  // GSW data is masked in non-water areas. Set it to 0 using unmask().
  // Invert the image to set all non-permanent region to 1
  var permanentWaterMask = permanentWater.unmask(0).not();
  
  var flooded = flooded.updateMask(permanentWaterMask);
  
  // Mask out areas with more than 5 degree slope using the SRTM DEM
  var slopeThreshold = 5;
  var slope = ee.Terrain.slope(srtm.clip(roi));
  var steepAreas = slope.gt(slopeThreshold);
  var slopeMask = steepAreas.not();
  Map.addLayer(slope, {min:0, max:1, palette:["cyan"]}, "Steep Areas", false);
  
  var flooded = flooded.updateMask(slopeMask);
  
  // --------------------------------------------------------------------------------
  //                             Remove Isolated Pixels
  // --------------------------------------------------------------------------------
  
  // connectedPixelCount is zoom dependent, so visual result will vary
  var connectedPixelThreshold = 20;
  var connections = flooded.connectedPixelCount(25);
  var disconnectedAreas = connections.lt(connectedPixelThreshold);
  var disconnectedAreaMask = disconnectedAreas.not();
  Map.addLayer(disconnectedAreas.selfMask(), {min:0, max:1, palette: ["yellow"]}, "Disconnected Areas", false);
  
  var flooded = flooded.updateMask(disconnectedAreaMask);
  Map.addLayer(flooded, {min:0, max:1, palette: ["red"]}, "Flooded Areas");
  
  