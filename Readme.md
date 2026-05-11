# Spectrum Analyzer

A Python package for spectral analysis of oceanographic tracer fields from Copernicus Marine model outputs.

The core idea is simple: the slope of the radial power spectrum carries information about the dominant dynamical regime of a region. A steep slope suggests large-scale circulation controls the tracer distribution; a shallower slope points to mesoscale or submesoscale dynamics. Spectrum Analyzer makes it easy to compute, visualise, and export these slopes so that different regions, depths, and tracers can be compared systematically.

## Current features

- Load tracer fields from Copernicus Marine NetCDF outputs
- Compute the 2D and radial power spectrum of a tracer field
- Estimate the spectral slope with a log-log regression
- Visualise the spectrum and the fitted slope
- Export spectrum and slope data for later comparison
## Planned Extension

- Slope estimation with uncertainty quantification
- Batch export for multi-region or multi-tracer comparison
- Support for additional tracers (salinity, temperature, chlorophyll, ...)
- Spectral analysis of time series
- Analysis at multiple depth levels
- Comparison with satellite observations

