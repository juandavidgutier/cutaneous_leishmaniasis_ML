# Read the data
leish <- read.csv("D:/clases/UDES/articulo leishmaniasis/shap/data_time_series.csv")

# Display structure of the data
str(leish)

# Convert the Cases column to a time series object
Cases <- ts(leish$Cases, start = c(2007,1), frequency = 12)

# Set graphical parameters for the plot
par(cex.lab = 1.8, cex.axis = 1.5, cex.main = 0.1)

# Plot the time series with specified text sizes
ts.plot(Cases, col = "red", xlab = "Time", ylab = "Cases", main = "")

# Reset graphical parameters to default after plotting (optional)
par(cex.lab = 1, cex.axis = 1, cex.main = 1)
